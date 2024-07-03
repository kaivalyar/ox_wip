python3 CODE/1_gen_embeddings.py
echo 'EMBEDDED'
echo '============================'

python3 CODE/2_prep_rot_inputs.py
echo 'PREPPED'
echo '============================'

python3 CODE/3_run_rot.py --epochs 5000 --batch_size 10000 --learning_rate 0.01
echo 'ROTed'
echo '============================'

python3 CODE/4_denorm.py
echo 'DeNormed'
echo '============================'

python3 CODE/5_viz.py
echo 'WordCloud Generated'
echo '============================'

python3 CODE/5.5_tokviz.py
echo 'Token Distributions Generated'
echo '============================'


python3 CODE/6_tokviz.py
echo 'Token Viz Generated'
echo '============================'

