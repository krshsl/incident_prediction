#! /bin/bash -l
#SBATCH --output=logfile

dir=/common/home/ks2025/rutgers/cs551/final_project
cd $dir

if [ ! -f TrafficEvents_Aug16_Dec20_Publish.tar.gz ]; then
    echo downloading traffic events data...
    curl "https://drive.usercontent.google.com/download?id=1IOTGHBPt-0cI8KgHYlwHT62OeAPKpOBc&export=download&confirm=t&uuid=f3a8b7b5-84b9-403b-ada4-e9e86a283795" -o "TrafficEvents_Aug16_Dec20_Publish.tar.gz"
fi

if [ ! -f TrafficEvents_Aug16_Dec20_Publish.csv ]; then
    tar -xzf TrafficEvents_Aug16_Dec20_Publish.tar.gz
fi

# if [ ! -f WeatherEvents_Aug16_Dec20_Publish.tar.gz ]; then
#     echo downloading weather events data...
#     curl "https://drive.usercontent.google.com/download?id=1WPWSW0yY5SLzmAYZeey4kA4iwY8Zwcce&export=download&confirm=t&uuid=3352431a-10a9-4154-8acd-31ce1ae3a637" -o "WeatherEvents_Aug16_Dec20_Publish.tar.gz"
# fi

# if [ ! -f WeatherEvents_Aug16_Dec20_Publish.csv ]; then
#     tar -xzf WeatherEvents_Aug16_Dec20_Publish.tar.gz
# fi

echo "generate output"
if [ ! -d "${dir}/.venv" ]
    then
        python3.12 -m venv .venv
        source .venv/bin/activate
        # pip --no-cache-dir uninstall tensorflow
        pip --no-cache-dir install 'tensorflow[and-cuda]'
        pip --no-cache-dir install -r requirements.txt
else
    source ${dir}/.venv/bin/activate
    # pip --no-cache-dir uninstall tensorflow
    pip --no-cache-dir install 'tensorflow[and-cuda]'
fi

cd src

echo "Input being created"
jupyter nbconvert --execute --to notebook --inplace 1_CreateInput.ipynb

echo "Output being predicted"
jupyter nbconvert --execute --to notebook --inplace 2_PredictEvent.ipynb

# pip --no-cache-dir uninstall tensorflow
deactivate
