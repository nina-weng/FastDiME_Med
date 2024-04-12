'''
evaluate shortcut learning

3 steps are included in the scripts:
* Train the classifiers based on the unbalanced datasets.
* Obtain the counterfactuals wrt. shortcut of the images in the commen test set.
* Evaluate the results and save as `.csv` and `.txt` file under `./records/shortcuts/`. 
'''

from shortcut_detection.train_biassed_classifier import main_train
from shortcut_detection.config import scdconfig,CLASSIFIERCONFIG
from shortcut_detection.exam_performance import main_sc_exam
from shortcut_detection.utils import *





def main(seed,
         scdconfig,
         isTrain=True,
         isExam = True):
    # train the 3 models for 3 datasets
    if isTrain:
        model_pth_list = main_train(
            seed,
            classifier_config=CLASSIFIERCONFIG
        )

    # exam the shortcurts learning

    
    if isExam:
        if isTrain:
            model_pth_list = get_ckpt(model_pth_list)
            print(f'{model_pth_list=}') 
            model_pth_list = model_pth_list
            scdconfig.model_path_dict = model_pth_list
            main_sc_exam(scdconfig,
                         model_pth_list)
        else:
            main_sc_exam(scdconfig)
            
            
        


if __name__ == '__main__':


    print('#'*30)
    print(f'the configs: \n{CLASSIFIERCONFIG=}\n{scdconfig=}')
    print('#'*30)

    isTrain = scdconfig.isTrain 
    isExam = scdconfig.isExam
    main(seed=42,
         scdconfig=scdconfig,
         isTrain=isTrain,
         isExam=isExam)