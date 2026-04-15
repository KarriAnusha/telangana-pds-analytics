# Telangana PDS Cluster Profile Report

## Run Summary
- Shops analyzed: 17,367
- K-Means clusters: 6
- DBSCAN outliers: 415
- Silhouette score (selected K): 0.4024
- Best K by silhouette curve: 2 (0.5713)
- Cluster purity: not computed (districtType column missing)

## Cluster Profiles

|   kmeans_cluster |   meanTransactions |   medianTransactions |   maxTransactions |   stdTransactions |   totalOtherShopTrans |   meanUtilization |   meanPortability |   meanTransactionToCardRatio |   totalRice |   totalWheat |   meanRiceWheatRatio |   totalRcs |   volatilityCoeff |   portabilityLoad |   seasonalPeakToMean |   shop_count | persona                    |
|-----------------:|-------------------:|---------------------:|------------------:|------------------:|----------------------:|------------------:|------------------:|-----------------------------:|------------:|-------------:|---------------------:|-----------:|------------------:|------------------:|---------------------:|-------------:|:---------------------------|
|                0 |            565.695 |              566.787 |           637.694 |           40.0607 |               91058.6 |            0.9103 |            0.5505 |                       0.9103 | 3.56281e+06 |    43685.6   |               2.505  |    633.444 |            0.079  |            0.5513 |               1.1417 |         4495 | Active Urban Shops         |
|                1 |            337.747 |              337.866 |           363.255 |           13.6954 |               13202.9 |            0.7971 |            0.136  |                       0.7971 | 2.0992e+06  |      249.488 |               0.1813 |    424.86  |            0.0453 |            0.1364 |               1.0844 |        11460 | Stable Rural Shops         |
|                2 |            329.298 |              329.643 |           351.643 |           12.86   |               12463.8 |           16.5897 |            0.1943 |                      16.5897 | 2.1293e+06  |        0     |               0      |    376.127 |            0.0412 |            0.1941 |               1.0742 |           14 | Anomalous High-Utilization |
|                3 |           1167.98  |             1171.83  |          1341.86  |           98.6044 |              264690   |            1.1066 |            0.7471 |                       1.1066 | 7.92945e+06 |   778722     |              12.5492 |   1077.43  |            0.0895 |            0.7472 |               1.1579 |         1099 | High-Volume Urban Hubs     |
|                4 |            196.905 |              185.109 |           384.255 |           93.9449 |               36412.5 |            0.4266 |            0.6378 |                       0.4266 | 1.14432e+06 |    46516.5   |               4.6616 |    470.801 |            0.5712 |            0.6289 |               2.2245 |          298 | Volatile Portability Hubs  |

## Top Suspicious Shops (Fraud Risk Proxy)

|   shopNo |   distCode | distName    |   kmeans_cluster |   meanTransactionToCardRatio |   clusterZScore |
|---------:|-----------:|:------------|-----------------:|-----------------------------:|----------------:|
|  2915019 |        926 | Kamareddy   |                1 |                       6.7772 |         42.2286 |
|  3101039 |        928 | Mahabubabad |                1 |                       2.5809 |         12.5967 |
|  1585761 |        537 | Ranga Reddy |                4 |                       2.1118 |          7.1775 |
|  3722021 |        934 | Sangareddy  |                0 |                       1.8702 |          4.9555 |
|  3727032 |        934 | Sangareddy  |                0 |                       1.8098 |          4.6432 |
|  1585676 |        537 | Ranga Reddy |                0 |                       1.7789 |          4.4839 |
|  2084165 |        534 | Karimnagar  |                0 |                       1.7397 |          4.2814 |
|  3313029 |        930 | Medchal     |                0 |                       1.7335 |          4.2497 |
|  3301037 |        930 | Medchal     |                1 |                       1.3793 |          4.1115 |
|  3302116 |        930 | Medchal     |                0 |                       1.6974 |          4.0631 |
|  3727049 |        934 | Sangareddy  |                0 |                       1.6867 |          4.0082 |
|  3727039 |        934 | Sangareddy  |                0 |                       1.684  |          3.994  |
|  3501032 |        932 | Nirmal      |                0 |                       1.6836 |          3.9918 |
|  3722006 |        934 | Sangareddy  |                0 |                       1.6799 |          3.973  |
|  3384226 |        930 | Medchal     |                0 |                       1.643  |          3.7823 |
|  3727033 |        934 | Sangareddy  |                0 |                       1.6421 |          3.778  |
|  3722031 |        934 | Sangareddy  |                0 |                       1.6308 |          3.7192 |
|  3611015 |        933 | Peddapalli  |                0 |                       1.6299 |          3.7148 |
|  3803023 |        935 | Siddipet    |                0 |                       1.6252 |          3.6907 |
|  2909032 |        926 | Kamareddy   |                0 |                       1.6229 |          3.6784 |

## Portability Hubs (Logistics Priority)

|   shopNo |   distCode | distName            |   kmeans_cluster |   portabilityLoad |   totalOtherShopTrans |
|---------:|-----------:|:--------------------|-----------------:|------------------:|----------------------:|
|  4406029 |        941 | Yadadri Bhuvanagiri |                4 |                 1 |                 13698 |
|  4411030 |        941 | Yadadri Bhuvanagiri |                4 |                 1 |                  8622 |
|  3402006 |        931 | Nagarkarnool        |                4 |                 1 |                    54 |
|  1677755 |        536 | Hyderabad           |                3 |                 1 |                284598 |
|  1803021 |        533 | nan                 |                1 |                 1 |                    16 |
|  1803022 |        533 | nan                 |                1 |                 1 |                    10 |
|  1803023 |        533 | nan                 |                1 |                 1 |                   113 |
|  1803017 |        533 | nan                 |                1 |                 1 |                    24 |
|  3612011 |        933 | Peddapalli          |                4 |                 1 |                 10458 |
|  1803018 |        533 | nan                 |                1 |                 1 |                     1 |
|  1803019 |        533 | nan                 |                1 |                 1 |                    41 |
|  1803020 |        533 | nan                 |                1 |                 1 |                     3 |
|  1581645 |        537 | Ranga Reddy         |                0 |                 1 |                200970 |
|  4406028 |        941 | Yadadri Bhuvanagiri |                4 |                 1 |                 17226 |
|  1820016 |        533 | nan                 |                1 |                 1 |                    62 |
|  1820017 |        533 | nan                 |                1 |                 1 |                    78 |
|  1820021 |        533 | nan                 |                4 |                 1 |                   151 |
|  1815028 |        533 | nan                 |                1 |                 1 |                     3 |
|  1815031 |        533 | nan                 |                1 |                 1 |                    67 |
|  1820012 |        533 | nan                 |                1 |                 1 |                     4 |