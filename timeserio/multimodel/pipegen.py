from keras.utils import Sequence


class _PipelineGenerator(Sequence):
    def __init__(self, *, x_pipes, y_pipes, df_generator):
        self.x_pipes = x_pipes
        self.y_pipes = y_pipes
        self.df_generator = df_generator

    def __len__(self):
        return len(self.df_generator)

    def __getitem__(self, item):
        df = self.df_generator[item]
        x = [pipe.transform(df) for pipe in self.x_pipes]
        y = [pipe.transform(df) for pipe in self.y_pipes]
        return x, y
