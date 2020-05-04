for h in {'100','100,100','50','50,50','50,100,50','50,50,50','200','200,200'}
do
nohup python baseline.py --label_type arousal --hidden_layers ${h} --audio_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_gemaps.nc --visual_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_facial_lm.nc audiovisual > logs/mlp_msp-improv_arousal_${h}_audiovisual_NOearlyStopping.log &
nohup python baseline.py --label_type valence --hidden_layers ${h} --audio_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_gemaps.nc --visual_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_facial_lm.nc audiovisual > logs/mlp_msp-improv_valence_${h}_audiovisual_NOearlyStopping.log &
nohup python baseline.py --label_type category --hidden_layers ${h} --audio_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_gemaps.nc --visual_data /mount/arbeitsdaten/slu/michael/data/msp-improv/msp-improv_facial_lm.nc audiovisual > logs/mlp_msp-improv_category_${h}_audiovisual_NOearlyStopping.log &
done
