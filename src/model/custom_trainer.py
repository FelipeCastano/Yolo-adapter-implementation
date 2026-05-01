from copy import deepcopy
import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from ultralytics.models.yolo.detect import DetectionTrainer
from YOLO_adapter import YOLOAdapterBlock
from adapter_manager import AdapterManager


class CustomTrainer(DetectionTrainer):
	"""Custom Ultralytics trainer that preserves adapter-only training.

	Subclasses :class:`~ultralytics.models.yolo.detect.DetectionTrainer` to:

	* Inject the pre-built adapted model, bypassing Ultralytics model
	  reconstruction from YAML.
	* Re-apply parameter freezing after the parent ``_setup_train`` unfreezes
	  all layers.
	* Run data-driven λ initialization using the training dataloader.
	* Rebuild the optimizer's parameter groups and learning-rate scheduler to
	  include only the trainable adapter parameters.
	* Save checkpoints with the correct adapted architecture instead of the
	  default Ultralytics 80-class serialization.

	:param adapted_model: The adapted YOLO model returned by
		:meth:`~adapter_manager.AdapterManager.build`.
	:type adapted_model: ultralytics.YOLO
	:param kwargs: Keyword arguments forwarded to
		:class:`~ultralytics.models.yolo.detect.DetectionTrainer`, including
		the ``overrides`` dict with training hyperparameters.
	"""

	def __init__(self, adapted_model, **kwargs):
		super().__init__(**kwargs)
		self._adapted_model = adapted_model.model

	def get_model(self, cfg=None, weights=None, verbose=True) -> torch.nn.Module:
		"""Return the pre-built adapted model instead of rebuilding from config.

		:param cfg: Ignored. Kept for API compatibility.
		:param weights: Ignored. Kept for API compatibility.
		:param verbose: Ignored. Kept for API compatibility.
		:return: The adapted detection model.
		:rtype: torch.nn.Module
		"""
		return self._adapted_model

	def _setup_train(self):
		"""Set up training with adapter-only parameter freezing.

		Calls the parent ``_setup_train`` (which builds the dataloader,
		optimizer, and scheduler), then:

		1. Re-freezes all parameters except those inside
		   :class:`~YOLO_adapter.YOLOAdapterBlock`.
		2. Runs data-driven λ initialization using the training dataloader.
		3. Filters the optimizer's parameter groups to include only trainable
		   parameters.
		4. Rebuilds the LR scheduler to match the reduced parameter groups.
		"""
		super()._setup_train()

		for param in self.model.parameters():
			param.requires_grad = False
		for module in self.model.modules():
			if isinstance(module, YOLOAdapterBlock):
				for param in module.parameters():
					param.requires_grad = True

		print("[CustomTrainer] Computing λ for data-driven initialization...")
		manager       = AdapterManager.__new__(AdapterManager)
		manager.model = type('_M', (), {'model': self.model})()
		manager._compute_lambda_and_init(self.train_loader, self.device)

		self.optimizer.param_groups = [
			g for g in self.optimizer.param_groups
			if any(p.requires_grad for p in g["params"])
		]

		self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)

		trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		total     = sum(p.numel() for p in self.model.parameters())
		print(f"[CustomTrainer] Trainable : {trainable:,} ({100 * trainable / total:.1f}%)")
		print(f"[CustomTrainer] Optimizer groups : {len(self.optimizer.param_groups)}")

	def save_model(self):
		"""Save checkpoints with the correct adapted architecture.

		Overrides the default Ultralytics serialization which stores the
		original 80-class ``DetectionModel``. Instead, saves the full adapted
		model — including :class:`~YOLO_adapter.C2f_Adapter` blocks and the
		detection head with the correct number of classes — so that
		:meth:`~adapter_manager.AdapterManager.load_trained` can restore it
		without any class mismatch.
		"""
		weights_dir = os.path.join(self.save_dir, 'weights')
		os.makedirs(weights_dir, exist_ok=True)

		self.ema.ema.names = self.model.names
		self.ema.ema.nc    = self.model.nc

		ckpt = {
			'epoch':        self.epoch,
			'best_fitness': self.best_fitness,
			'model':        deepcopy(self.ema.ema).half(),
			'optimizer':    None,
			'train_args':   vars(self.args),
			'date':         None,
		}

		last = os.path.join(weights_dir, 'last.pt')
		torch.save(ckpt, last)

		if self.best_fitness == self.fitness:
			best = os.path.join(weights_dir, 'best.pt')
			torch.save(ckpt, best)
			print(f"[CustomTrainer] Best model saved → {best}")