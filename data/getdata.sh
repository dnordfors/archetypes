#!/bin/sh
# Import O*NET database
curl -O https://www.onetcenter.org/dl_files/database/db_23_2_excel.zip
# Import Census ACS/:UMS data
curl -O https://www2.census.gov/programs-surveys/popest/geographies/2016/state-geocodes-v2016.xls
curl -O https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.csv
curl -O https://www2.census.gov/programs-surveys/acs/data/pums/2017/5-Year/csv_pca.zip
