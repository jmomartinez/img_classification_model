from model_module import img_model
import preprocessing_module

def main():
    #Fashion_mnist or mnist dataset
    # model = img_model(6,(28,28,1),(2,2),(3,3), None,'mnist','normalize')

    #Brain tumor dataset
    path = '../../../Datasets/brain_tumor_dataset'

    model = img_model(6,(386,354,1),(2,2),(3,3),path,'path','normalize')
    
    model.predict()

if __name__=='__main__':
    main()