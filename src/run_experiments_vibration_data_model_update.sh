echo "random model: FC first layer 8, 2 hidden layers with 20 each"
python nn_trainer.py -train ../data/siemens_data/HealthyTool_vibrationData_50kHz_train.npy -val ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -model_name FC_siemens -model_params 8 2 20 -epochs 0 -model_file modelfiles/modelfile_random_first_8_2hidden_20
echo "no model update"
echo "0.01 error"
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_random_first_8_2hidden_20_no_model_update_0.01.7z -model_file modelfiles/modelfile_random_first_8_2hidden_20 -maxerror 0.01
echo "0.1 error"
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_random_first_8_2hidden_20_no_model_update_0.1.7z -model_file modelfiles/modelfile_random_first_8_2hidden_20 -maxerror 0.1
