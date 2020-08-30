from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(r'')
model_trainer.trainModel(num_objects=42, num_experiments=50, enhance_data=True, batch_size=32, show_network_summary=True,transfer_from_model="resnet50_weights_tf_dim_ordering_tf_kernels.h5", initial_num_objects=1000)