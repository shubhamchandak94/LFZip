#!/bin/bash
echo "trained model: FC first layer 32, 4 hidden layers with 128 each"
#python nn_trainer.py -train ../data/siemens_data/HealthyTool_vibrationData_50kHz_train.npy -val ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -model_name FC_siemens -model_params 32 4 128 -epochs 0 -model_file modelfiles/modelfile_random_first_32_4hidden_128
#echo "no model update"
#echo "0.01 error"
#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_random_first_8_2hidden_20_no_model_update_0.01.7z -model_file modelfiles/modelfile_random_first_8_2hidden_20 -maxerror 0.01
#echo "0.1 error"
#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_random_first_8_2hidden_20_no_model_update_0.1.7z -model_file modelfiles/modelfile_random_first_8_2hidden_20 -maxerror 0.1
#
for model_update_period in 400 1000
do
    for lr in 0.0005 0.001 0.004 0.008 
    do
        for num_epochs in 1 4 16
        do 
            echo "model update period" $model_update_period
            echo "lr" $lr
            echo "num_epochs" $num_epochs
            maxerror=0.01
            echo "maxerror" $maxerror
            CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_first_32_4hidden_128_wnoise_0.01_model_update_period"_"$model_update_period"_"lr"_"$lr"_"num_epochs"_"$num_epochs"_"$maxerror.7z -model_file modelfiles/modelfile_trained_vibration_first_32_4hidden_128_noise_0.01 -maxerror $maxerror -model_update_period $model_update_period -lr $lr -epochs $num_epochs
            maxerror=0.1
            echo "maxerror" $maxerror
            CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode c -infile ../data/siemens_data/HealthyTool_vibrationData_50kHz_val.npy -outfile vibration_val_compressed/vibration_val_trained_first_32_4hidden_128_wnoise_0.05_model_update_period"_"$model_update_period"_"lr"_"$lr"_"num_epochs"_"$num_epochs"_"$maxerror.7z -model_file modelfiles/modelfile_trained_vibration_first_32_4hidden_128_noise_0.05 -maxerror $maxerror -model_update_period $model_update_period -lr $lr -epochs $num_epochs
                # decompress and check
    #            echo "verifying decompression"
    #            CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python nn_compress.py -mode d -infile vibration_val_compressed/vibration_val_trained_random_first_16_2hidden_40_model_update_period"_"$model_update_period"_"lr"_"$lr"_"num_epochs"_"$num_epochs"_"$maxerror.7z -outfile cmpcmp.npy -model_file modelfiles/modelfile_random_first_16_2hidden_40
    #            cmp vibration_val_compressed/vibration_val_trained_random_first_16_2hidden_40_model_update_period"_"$model_update_period"_"lr"_"$lr"_"num_epochs"_"$num_epochs"_"$maxerror.7z.recon.npy cmpcmp.npy
        done
    done
done

