from models.nn_models import NN_Model
from ..config_creator import get_config
import torch
from models.helper_functions import fill_default_key_conf, fill_default_key, create_actions, merge_state_actions


OUTPUT_DIMS = {'qoe': 1, 'ssim': 3, 'bit_rate': 3, 'all': 5}


class SLModel(NN_Model):
    def __init__(self, model_config):
       super(SLModel, self).__init__(get_config()['nn_input_size'], OUTPUT_DIMS[fill_default_key_conf(model_config, 'scoring_function_type')])
       self.CONFIG = get_config()
       self.scoring_type = fill_default_key_conf(model_config, 'scoring_function_type')
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.output_size = OUTPUT_DIMS[self.scoring_type]
       self.actions = create_actions()
       self.model_name = fill_default_key(model_config, 'sl_model_name', f"sl_weights_abr_{get_config()['abr']}_{self.scoring_type}.pt")
       self.model_config = model_config
    
    def calc_score(self, state, action):
        x = self.model(merge_state_actions(state['state'], action))
        if self.output_size == 1:
            return x
        if self.output_size == 3:
            return x[:, 0] - self.change_coef * x[:, 1] - self.buffer_coef * x[:, 2]
        return x[:, 3] - self.change_coef * x[:, 4] - self.buffer_coef * x[:, 2]

    def predict(self, sent_state):
        best_action = -1
        best_score = 0
        for i, action in enumerate(self.actions):
            score = self.calc_score(sent_state, action)
            if best_action == -1 or best_score < score:
                best_action, best_score = i, score
        return best_action

    def load(self):
        dct = torch.load(fill_default_key_conf(self.model_config, 'weights_path') + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path=''):
        if get_config()['test']:
            return
        if path == '':
            path = self.model_name
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{fill_default_key_conf(self.model_config, 'weights_path')}{path}")
    
    def done(self):
        pass