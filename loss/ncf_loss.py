from loss.loss_class import Loss
import torch
import torch.nn.functional as F

class NCFLoss(Loss):
    
    def __init__(self, name='NCFLoss'):
        """
        weighted_vector: vector of weights of every item, default None. Could be for example the prices of
                         every item (price_vector) or novelty_vector etc.
        """
        super().__init__(name)
    def compute_loss(self, y_true, y_pred,  weighted_vector=None):
        

        # check if good dimensions between y_pred and y_true
        # self.__check_dim_pred_gt__(y_pred, y_true)
        # # check if mean or log_variance are not None
        # self.__check_is_mean_var__(mean, log_variance)

        # calculate the reconstruction loss
        if(weighted_vector is not None):
            loss = -weighted_vector*(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
            return loss.mean()
            
        else:
            loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
            return loss.mean()