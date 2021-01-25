from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.squeeze import get_squeezer_by_name
from utils.parameter_parser import parse_params

class FeatureSqueezingRC:
    def __init__(self, keras_model, rc_name):
        # Example of rc_name: FeatureSqueezing?squeezer=bit_depth_1
        self.model_predict = lambda x: keras_model(torch.as_tensor(x).to("cuda:0")).detach().cpu().numpy()

        subject, params = parse_params(rc_name)
        assert subject == 'FeatureSqueezing'

        if ('squeezer') in params:
            self.filter = get_squeezer_by_name(params['squeezer'], 'python')
        elif 'squeezers' in params:
            squeezer_names = params['squeezers'].split(',')
            self.filters = [ get_squeezer_by_name(squeezer_name, 'python') for squeezer_name in squeezer_names ]

            def filter_func(x, funcs):
                x_p = x
                for func in funcs:
                    x_p = func(x_p)
                return x_p

            self.filter = lambda x: filter_func(x, self.filters)



    def predict(self, X):
        import numpy as np

        X_filtered = self.filter(X)
        if X_filtered.shape[1] != 3 and X_filtered.shape[2] != 28:
            X_filtered = np.moveaxis(X_filtered, 3, -3)

        preds = []
        for i in range(0, len(X_filtered), 64):
            preds.append(self.model_predict(X_filtered[i:i + 64]))
        Y_pred = np.vstack(preds)
        return Y_pred

    def visualize_and_predict(self, X):
        import numpy as np
        X_filtered = self.filter(X)

        if X_filtered.shape[1] != 3 and X_filtered.shape[2] != 28:
            X_filtered = np.moveaxis(X_filtered, 3, -3)

        preds = []
        for i in range(0, len(X_filtered), 64):
            preds.append(self.model_predict(X_filtered[i:i + 64]))
        Y_pred = np.vstack(preds)
        return X_filtered, Y_pred