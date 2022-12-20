python entrypoint.py infer \
--model_filepath ./model/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000002/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--scale_parameters_filepath ./scale_params.npy


sudo singularity build cyber-pdf-dec2022_sts_SRI_weight_grad_v1.simg trojan_detector.def 


singularity run \
--bind /work2/project/trojai-cyber-pdf \
--nv \
./cyber-pdf-dec2022_sts_SRI_weight_grad_v1.simg \
infer \
--model_filepath=./model/id-00000002/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./model/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/ \
--scale_parameters_filepath ./scale_params.npy