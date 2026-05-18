# Telangana PDS Cluster Profile Report

## Run Summary
- Shops analyzed: 17,367
- K-Means clusters: 5
- DBSCAN outliers: 537
- Silhouette score (selected K): 0.3746
- Best K by silhouette curve: 2 (0.5436)
- Note: K=5 is retained for operationally useful shop personas, even though the silhouette curve prefers a different K.
- Cluster purity: not computed (districtType column missing)

## Cluster Profiles

| kmeans_cluster | meanTransactions | medianTransactions | maxTransactions | stdTransactions | totalOtherShopTrans | meanUtilization | meanPortability | meanTransactionToCardRatio | totalRice | totalWheat | meanRiceWheatRatio | totalRcs | volatilityCoeff | portabilityLoad | seasonalPeakToMean | shop_count | persona |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1178.9148 | 1183.007 | 1351.1502 | 101.5374 | 14918.7108 | 1.1334 | 0.7508 | 1.1334 | 444757.7155 | 43523.6601 | 7.5766 | 1065.5996 | 0.0906 | 0.7509 | 1.1532 | 1065 | High-Volume Urban Hubs |
| 1 | 339.409 | 339.5501 | 364.9148 | 14.0974 | 738.475 | 0.8 | 0.1319 | 0.8 | 117483.765 | 19.2157 | 0.1875 | 427.6764 | 0.0449 | 0.1324 | 1.081 | 11218 | Stable Rural Shops |
| 2 | 238.6866 | 233.9374 | 369.8693 | 69.6859 | 2360.2458 | 0.4856 | 0.5962 | 0.4856 | 81240.4816 | 2979.7944 | 3.546 | 474.5913 | 0.3318 | 0.5939 | 1.6568 | 895 | Volatile Portability Hubs |
| 3 | 592.5525 | 593.8159 | 658.4202 | 37.9286 | 5165.1586 | 0.9476 | 0.5331 | 0.9476 | 207485.1519 | 2415.844 | 2.3299 | 638.5106 | 0.0672 | 0.5338 | 1.1155 | 4186 | Active Urban Shops |
| 4 | 1225.4902 | 1232.0 | 1406.0 | 94.1529 | 13944.3333 | 1.0278 | 0.6771 | 1.0278 | 460284.3333 | 11874.6667 | 1806.101 | 1168.098 | 0.0806 | 0.6772 | 1.1403 | 3 | High-Volume Urban Hubs |

## Top Suspicious Shops (Fraud Risk Proxy)

| shopNo | distCode | distName | kmeans_cluster | meanTransactionToCardRatio | clusterZScore |
| --- | --- | --- | --- | --- | --- |
| 3722021 | 934 | Sangareddy | 3 | 1.8702 | 5.7712 |
| 3727032 | 934 | Sangareddy | 3 | 1.8103 | 5.3961 |
| 1585676 | 537 | Ranga Reddy | 3 | 1.7789 | 5.1997 |
| 2084165 | 534 | Karimnagar | 3 | 1.7424 | 4.9717 |
| 3313029 | 930 | Medchal | 3 | 1.7348 | 4.924 |
| 3302116 | 930 | Medchal | 3 | 1.7008 | 4.7115 |
| 3501032 | 932 | Nirmal | 3 | 1.6968 | 4.6863 |
| 3727049 | 934 | Sangareddy | 3 | 1.6867 | 4.6234 |
| 3384226 | 930 | Medchal | 3 | 1.6464 | 4.3708 |
| 3727033 | 934 | Sangareddy | 3 | 1.6423 | 4.3455 |
| 3611015 | 933 | Peddapalli | 3 | 1.6304 | 4.2709 |
| 3803023 | 935 | Siddipet | 3 | 1.6289 | 4.2618 |
| 3719001 | 934 | Sangareddy | 3 | 1.6211 | 4.2128 |
| 3719041 | 934 | Sangareddy | 3 | 1.6159 | 4.18 |
| 1585770 | 537 | Ranga Reddy | 3 | 1.6096 | 4.1407 |
| 3719038 | 934 | Sangareddy | 3 | 1.6042 | 4.1073 |
| 3719046 | 934 | Sangareddy | 3 | 1.6017 | 4.0916 |
| 3727042 | 934 | Sangareddy | 3 | 1.5979 | 4.0677 |
| 3302118 | 930 | Medchal | 3 | 1.5929 | 4.0362 |
| 2357014 | 539 | Nalgonda | 3 | 1.5844 | 3.9833 |

## Portability Hubs (Logistics Priority)

| shopNo | distCode | distName | kmeans_cluster | portabilityLoad | totalOtherShopTrans |
| --- | --- | --- | --- | --- | --- |
| 4406029 | 941 | Yadadri Bhuvanagiri | 2 | 1.0 | 761 |
| 4411030 | 941 | Yadadri Bhuvanagiri | 2 | 1.0 | 479 |
| 3402006 | 931 | Nagarkarnool | 2 | 1.0 | 3 |
| 1677755 | 536 | Hyderabad | 0 | 1.0 | 15811 |
| 1803021 | 533 |  | 2 | 1.0 | 16 |
| 1803022 | 533 |  | 2 | 1.0 | 10 |
| 1803023 | 533 |  | 2 | 1.0 | 113 |
| 1803017 | 533 |  | 2 | 1.0 | 24 |
| 3612011 | 933 | Peddapalli | 2 | 1.0 | 581 |
| 1803018 | 533 |  | 2 | 1.0 | 1 |
| 1803019 | 533 |  | 2 | 1.0 | 41 |
| 1803020 | 533 |  | 2 | 1.0 | 3 |
| 1581645 | 537 | Ranga Reddy | 3 | 1.0 | 11165 |
| 4406028 | 941 | Yadadri Bhuvanagiri | 2 | 1.0 | 957 |
| 1820016 | 533 |  | 2 | 1.0 | 62 |
| 1820017 | 533 |  | 2 | 1.0 | 78 |
| 1820021 | 533 |  | 2 | 1.0 | 151 |
| 1815028 | 533 |  | 2 | 1.0 | 3 |
| 1815031 | 533 |  | 2 | 1.0 | 67 |
| 1820012 | 533 |  | 2 | 1.0 | 4 |