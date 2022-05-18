from jax_python.aad import AbstractAbnormalDetect
from spark_lca_train.Train import Train

__version__="2.0"


class LcaTrain(AbstractAbnormalDetect):

	def __init__(self):
		super(LcaTrain, self).__init__()
		self.config = {}

	def version(self):
		return __version__

	def configure(self, config):
		self.train = Train.from_map(config)
		self.config = config

	def transform(self, *df):
		df = df[0]

		result = self.train.run(df)
		return result

	def fields(self):
		return [('is_sparse', 'bool'), ('is_period', 'bool'), ('per_cor', 'int'),
				('per_coef', 'float'), ('is_accidental', 'bool')]

	def mode_append(self):
		return True