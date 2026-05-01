import torch
import torch.nn as nn
import ultralytics.nn.modules.block as block_module
from ultralytics import YOLO
from YOLO_adapter import YOLOAdapterBlock, C2f_Adapter


class AdapterManager:
	"""Manages the full lifecycle of YOLO-Adapter model construction and weight loading.

	Handles C2f block replacement, layer freezing, data-driven weight
	initialization, and checkpoint loading. Designed to be model-size agnostic:
	the expansion ratio ``r`` is assigned automatically based on each C2f
	block's parameter count.

	:param pt_path: Path to the base YOLOv8 ``.pt`` checkpoint.
	:type pt_path: str

	.. rubric:: Expansion ratio thresholds

	* ``N_p > 5M`` → ``r = 1.0`` (Heavy)
	* ``1M < N_p ≤ 5M`` → ``r = 0.5`` (Medium)
	* ``N_p ≤ 1M`` → ``r = 0.25`` (Light)
	"""

	HEAVY_THRESHOLD  = 5_000_000
	MEDIUM_THRESHOLD = 1_000_000

	def __init__(self, pt_path: str):
		self.pt_path = pt_path
		self.model   = None

	def build(self, nc: int = None) -> YOLO:
		"""Build the adapted model from the base checkpoint.

		Replaces all C2f blocks with :class:`~YOLO_adapter.C2f_Adapter`,
		copies the original C2f weights, freezes the backbone, and prints
		a parameter summary.

		:param nc: Number of output classes. If ``None``, keeps the original
			checkpoint value (80 for COCO). When provided, the detection head
			is reinitialised with the new class count.
		:type nc: int, optional
		:return: The adapted YOLO model ready for training.
		:rtype: YOLO
		"""
		self.model = YOLO(self.pt_path)

		if nc is not None and nc != self.model.model.nc:
			print(f"[AdapterManager] Adjusting nc: {self.model.model.nc} → {nc}")
			self.model.model.nc = nc
			self.model.model.yaml['nc'] = nc

		self._replace_c2f_blocks()
		self._freeze_backbone()
		self._print_summary()
		return self.model

	def load_trained(self, weights_path: str, nc: int = None) -> YOLO:
		"""Rebuild the adapted architecture and load fine-tuned weights.

		Useful for inference: rebuilds the custom architecture (required since
		Ultralytics cannot deserialize :class:`~YOLO_adapter.C2f_Adapter`
		from a plain checkpoint without the custom architecture being present)
		and then loads the trained state dict.

		:param weights_path: Path to the fine-tuned ``.pt`` checkpoint produced
			by :class:`~custom_trainer.CustomTrainer`.
		:type weights_path: str
		:param nc: Number of output classes. Must match the value used during
			training. If ``None``, keeps the original checkpoint value (80).
		:type nc: int, optional
		:return: The adapted YOLO model with trained weights loaded.
		:rtype: YOLO
		"""
		self.model     = self.build(nc=nc)
		checkpoint     = torch.load(weights_path, map_location='cpu', weights_only=False)
		self.model.model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
		self.model.model.names = checkpoint['model'].names
		self.model.model.nc    = checkpoint['model'].nc
		return self.model

	def init_adapters_from_data(self, dataloader, device: str = 'cpu'):
		"""Run data-driven adapter weight initialization.

		Performs a single forward pass over the full dataloader to compute
		the variance ratio λ for each :class:`~YOLO_adapter.C2f_Adapter` block,
		then reinitialises the adapter weights with ``N(0, 2λ / (n_in + n_out))``.
		Must be called after :meth:`build`.

		:param dataloader: DataLoader yielding batches with an ``'img'`` key
			containing uint8 tensors of shape ``(B, 3, H, W)``.
		:param device: Torch device string, e.g. ``'cpu'`` or ``'cuda:0'``.
		:type device: str
		:raises AssertionError: If :meth:`build` has not been called first.
		"""
		assert self.model is not None, "Call build() before init_adapters_from_data()"
		self._compute_lambda_and_init(dataloader, device)

	# ── Private ───────────────────────────────────────────────────────────────

	def _get_expansion_ratio(self, n_params: int) -> float:
		"""Return the adapter expansion ratio based on block parameter count.

		:param n_params: Total number of parameters in the C2f block.
		:type n_params: int
		:return: Expansion ratio ``r`` in ``{0.25, 0.5, 1.0}``.
		:rtype: float
		"""
		if n_params > self.HEAVY_THRESHOLD:
			return 1.0
		elif n_params > self.MEDIUM_THRESHOLD:
			return 0.5
		return 0.25

	def _replace_c2f_blocks(self):
		"""Replace every C2f block in the model with a C2f_Adapter.

		Copies the original C2f weights using ``strict=False`` and transfers
		the Ultralytics graph metadata attributes (``f``, ``i``, ``type``)
		required by ``_predict_once``.
		"""
		for name, module in list(self.model.model.named_modules()):
			if type(module) is block_module.C2f:
				parent_name, child_name = name.rsplit(".", 1)
				parent   = self.model.model.get_submodule(parent_name)
				c1       = module.cv1.conv.in_channels
				c2       = module.cv2.conv.out_channels
				n        = len(module.m)
				n_params = sum(p.numel() for p in module.parameters())
				r        = self._get_expansion_ratio(n_params)
				adapter  = C2f_Adapter(c1, c2, n=n, r=r)
				adapter.load_state_dict(module.state_dict(), strict=False)
				adapter.f    = module.f
				adapter.i    = module.i
				adapter.type = module.type
				setattr(parent, child_name, adapter)
				print(f"  {name}: N_p={n_params:,}  r={r}  c_hidden={max(1, int(c1 * r))}")

	def _freeze_backbone(self):
		"""Freeze all model parameters except those inside YOLOAdapterBlock."""
		for param in self.model.model.parameters():
			param.requires_grad = False
		for module in self.model.model.modules():
			if isinstance(module, YOLOAdapterBlock):
				for param in module.parameters():
					param.requires_grad = True

	def _compute_lambda_and_init(self, dataloader, device: str):
		"""Compute λ for each adapter and reinitialise its weights.

		Registers forward hooks on every :class:`~YOLO_adapter.C2f_Adapter`
		to record input and output variance statistics across the full
		dataloader, then calls :meth:`_init_adapter_weights` with the computed λ.

		:param dataloader: DataLoader yielding batches with an ``'img'`` key.
		:param device: Torch device string.
		:type device: str
		"""
		self.model.model.eval()
		hooks = []
		stats = {}

		for name, module in self.model.model.named_modules():
			if isinstance(module, C2f_Adapter):
				stats[name] = {'in': [], 'out': []}

				def make_hook(n):
					def hook(mod, inp, out):
						stats[n]['in'].append(inp[0].detach().var(dim=[0, 2, 3]).mean().item())
						stats[n]['out'].append(out.detach().var(dim=[0, 2, 3]).mean().item())
					return hook

				hooks.append(module.register_forward_hook(make_hook(name)))

		with torch.no_grad():
			for batch in dataloader:
				imgs = batch['img'].to(device).float() / 255.0
				self.model.model(imgs)

		for h in hooks:
			h.remove()

		for name, module in self.model.model.named_modules():
			if isinstance(module, C2f_Adapter) and name in stats:
				mean_var_in  = sum(stats[name]['in'])  / len(stats[name]['in'])
				mean_var_out = sum(stats[name]['out']) / len(stats[name]['out'])
				lam          = mean_var_out / (mean_var_in + 1e-8)
				self._init_adapter_weights(module.adapter, lam)
				print(f"  [λ-init] {name}: λ={lam:.4f}")

		self.model.model.train()

	@staticmethod
	def _init_adapter_weights(adapter_block: YOLOAdapterBlock, lam: float):
		"""Initialise adapter weights with the data-driven Xavier strategy.

		Each convolutional layer is initialised from
		``N(0, sqrt(2λ / (n_in + n_out)))``, where ``n_in`` and ``n_out``
		are the fan-in and fan-out of the layer, and λ is the variance ratio
		of the parallel C2f block computed from a forward pass over the
		training data. Bias terms are initialised to zero.

		:param adapter_block: The :class:`~YOLO_adapter.YOLOAdapterBlock`
			whose weights will be reinitialised.
		:type adapter_block: YOLOAdapterBlock
		:param lam: Variance ratio λ = E[Var(X_out)] / E[Var(X_in)] computed
			from the parallel C2f block.
		:type lam: float
		"""
		for layer in [adapter_block.conv2d, adapter_block.conv_transpose]:
			n_in  = layer.in_channels  * layer.kernel_size[0] * layer.kernel_size[1]
			n_out = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
			std   = (2 * lam / (n_in + n_out)) ** 0.5
			nn.init.normal_(layer.weight, mean=0.0, std=std)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)

	def _print_summary(self):
		"""Print a summary of trainable vs frozen parameters."""
		total      = sum(p.numel() for p in self.model.model.parameters())
		trainable  = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
		n_adapters = sum(1 for m in self.model.model.modules() if isinstance(m, C2f_Adapter))
		print(f"\nC2f_Adapter blocks inserted : {n_adapters}")
		print(f"Total parameters            : {total:,}")
		print(f"Trainable parameters        : {trainable:,}  ({100 * trainable / total:.1f}%)")
		print(f"Frozen parameters           : {total - trainable:,}  ({100 * (total - trainable) / total:.1f}%)")