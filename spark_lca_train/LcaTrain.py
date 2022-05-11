from jax_python.aad import AbstractAbnormalDetect
from spark_lca_train.Train import Train

__version__="2.1"


class LcaTrain(AbstractAbnormalDetect):

	def __init__(self):
		super(LcaTrain, self).__init__()
		self.config = {}

	def version(self):
		return __version__

	def configure(self, config):
		self.train = Train.from_map(config)
		self.config = config

	def transform(self, value_his):   # TODO
		result = self.train.run(value_his)
		return result

	def fields(self):   # TODO
		return [('pre_value', 'float'), ('upper', 'float'), ('lower', 'float'),
				('anomaly', 'int'), ('abnormality', 'int')]

	def mode_append(self):
		return True