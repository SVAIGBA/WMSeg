# SIGHAN2005

# download SIGHAN2005 from http://sighan.cs.uchicago.edu/bakeoff2005/
wget http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip

unzip icwb2-data.zip
rm icwb2-data.zip

# process SIGNHAN2005

python data_preprocessing.py --dataset=sighan2005 --translate

# CTB6

# please obtain CTB6 https://catalog.ldc.upenn.edu/LDC2007T36

# process CTB6
python data_preprocessing.py --dataset=ctb6
