#/usr/bin/env bash
#!/home/lhq323/1Tdev/yangrx/miniconda3/envs/myenv1 bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t ocelot23algo "$SCRIPTPATH"
