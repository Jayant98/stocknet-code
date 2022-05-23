#!/bin/bash
P="/content/stocknet-BERT-orig";
echo "Starting run #1";
rm -rf $P/src/config.yml;
cp $P/configs/config1.yml $P/src/config.yml;
python $P/src/Main.py;
echo "Starting run #2";
rm $P/src/config.yml;
cp $P/configs/config2.yml $P/src/config.yml;
python $P/src/Main.py;
echo "Starting run #3";
rm $P/src/config.yml;
cp $P/configs/config3.yml $P/src/config.yml;
python $P/src/Main.py;
echo "Starting run #4";
rm $P/src/config.yml;
cp $P/configs/config4.yml $P/src/config.yml;
python $P/src/Main.py;