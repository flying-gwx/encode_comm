conda activate tf
TXT_PARENT_PATH=/home/lhr1/database/wspsnr_res/
RESULT_PATH=/home/lhr1/database/GCN_data/all_csv/origin_csv
ALL_FILES=$(ls $TXT_PARENT_PATH)
#PROBLEM_FILE=('15to16_MC'  '15to16_StarWars'  '15to16_RingMan'  '5to6_LetsNotBeAloneTonight'  '15to16_LetsNotBeAloneTonight')
# 现在剩下两个problem_file ('15to16_StarWars', '15to16_RingMan')
PROBLEM_FILE=('15to16_StarWars'  '15to16_RingMan' )
# 还有个文件 5to6_WesternSichuan 都没有创立
#TXT_LIST=('15to16_BFG'  '15to16_BTSRun'  '15to16_BlueWorld'  '15to16_CMLauncher'  '15to16_CS'  '15to16_CandyCarnival')
for i in ${ALL_FILES[*]}
do
    flag=true
    for j in ${PROBLEM_FILE[*]}
    do
        if test $i == $j; then
            echo "$i is not correct!"
            flag=false
            break
        fi
    done
if test $flag == true; then
    python make_csv.py --txts_path $TXT_PARENT_PATH$i --result_path $RESULT_PATH
fi
done