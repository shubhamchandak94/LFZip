for noise in "0.0" "0.005" "0.01" "0.02" "0.05" "0.1" "0.2"
do
    python nn_trainer.py -train ../data/siemens_data/HealthyTool_vibrationData_50kHz_train.npy -val ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -model_name biGRU_siemens -epochs 20 -model_file modelfile_trained_vibration_first_32_biGRU_noise_$noise -noise $noise
    python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_trained_first_32_biGRU_wnoise_$noise"_0.01.7z" -model_file modelfile_trained_vibration_first_32_biGRU_noise_$noise -maxerror 0.01
    python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_trained_first_32_biGRU_wnoise_$noise"_0.1.7z" -model_file modelfile_trained_vibration_first_32_biGRU_noise_$noise -maxerror 0.1
done
