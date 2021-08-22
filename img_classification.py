from model_module import basic_cnn, resNet, efficientNet
import preprocessing_module

def main():
    path = '../../../Datasets/brain_tumor_dataset'
    # Fashion_mnist or mnist dataset
    # model = img_model(6,(28,28,1),(2,2),(3,3), None,'mnist','normalize')
    # model = resNet(6,(386,354,3),(2,2),(3,3),path,'path','normalize',False)
    # model = efficientNet(6,(386,354,3),(2,2),(3,3),path,'path','normalize',False)


    # Brain tumor dataset
    # model = basic_cnn(6,(386,354,1),(2,2),(3,3),path,'path','normalize')
    model = resNet(6,(386,354,3),(2,2),(3,3),path,'path','normalize',False)
    # model = efficientNet(6,(386,354,3),(2,2),(3,3),path,'path','normalize',False)


    model.predict()

if __name__=='__main__':
    main()