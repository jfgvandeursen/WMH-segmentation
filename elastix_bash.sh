#!/bin/bash

### paths ###
p_file="/home/jdvandeursen/Downloads/wmhsegElastix.txt" # path file path
o_dir="/home/jdvandeursen/Downloads/elastixOutput2/" # output folder path
participants="/home/jdvandeursen/Downloads/batch1/T1/participants.tsv" # participants.tsv path
unique_paths=true # true for unique output paths and false for using the o_dir only
# !!! change other paths on line 31 !!! #
######

echo "Starting..."

i=0
while read id other || [ -n "$id" ]; do # reading file line by line, even last line if it doesn't have a newline at the end ('a' contains only first column and 'b' are the other ones. If correct number of variables are present (e.g. a b c for 3 columns) every variable will contain the corresponding element.
    ((i=i+1))
    if ! [[ $((i)) == 1 ]]; then # if the line is not the first one (beacuse it contains the headers, we can skip that)
        echo "Processing '${id}'..."
		#process $id $p_file $o_dir # run the process funtion with parameter $1: id, $2: parameter file path, $3: output path
		if $unique_paths; then
			mkdir "${o_dir}/${id}"
			output="${o_dir}/${id}"
		fi
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test-jente
#SBATCH --time=1:00:00
#SBATCH --partition=rng-short
module purge
module load elastix/4.8
elastix -f "/home/jdvandeursen/Downloads/batch1/T1/sub-${id}/anat/sub-${id}_T1w.nii.gz" -m "/home/jdvandeursen/Downloads/batch1/FLAIR/sub-${id}/anat/sub-${id}_FLAIR.nii.gz" -p "${p_file}" -out "${output}"
EOT
		echo "${id} processed."
    fi
done < "$participants"
