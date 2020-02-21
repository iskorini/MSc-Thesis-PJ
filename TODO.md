## Step iniziale
---------------------------------------------------------------------------------------------------------------------------------

- [x] 1 ADDESTRAMENTO PARTITO DA COCO SU VISIBILE IN KAIST_MPD: annotated_test_visible

- [x] 2 ADDESTRAMENTO PARTITO DAI PESI DI 1 SU KAIST_MPD TERMICO: annotated_test_LWIR -> LWIR_KAIST_FROM_VIS_43.h5
  - 53443 instances of class person with average precision: 0.3663
  - 1396 instances of class cyclist with average precision: 0.0392
  - mAP using the weighted average of precisions among classes: 0.3580
  - mAP: 0.2028 
GIORNO:
   - 38802 instances of class person with average precision: 0.3415
   - 818 instances of class cyclist with average precision: 0.0387
   - mAP using the weighted average of precisions among classes: 0.3353
   - mAP: 0.1901
NOTTE:
   - 14641 instances of class person with average precision: 0.4440
   - 578 instances of class cyclist with average precision: 0.0440
   - mAP using the weighted average of precisions among classes: 0.4288
   - mAP: 0.2440 -> scritto

- [x] TEST DEI PESI DI 2 SU FLIR TERMICO: annotated_test_FLIR (testo i pesi di kaist su flir) -> scritto 

- [x] ADDESTRAMENTO PARTITO DAI PESI DI 2 SU FLIR TERMICO: ANNOTATED_FLIR_TRAINED -> scritto train da scrivere test


- [x] 5 ADDESTRAMENTO PARTITO DAI PESI DI COCO SU FLIR TERMICO: FLIR_FROM_COCO -> scritto


- [x] TEST DEI PESI DI 5 SU KAIST TERMICO: KAIST_FROM_FLIR
nohup retinanet-evaluate --score-threshold 0.30 --save-path ./KAIST_FROM_FLIR  --convert-model --gpu 0 csv KAIST_MPD/imageSets/csv_files_NO_PEOPLE/lwir/test-all-01.csv KAIST_MPD/class_name_to_id_NO_PEOPLE.csv  weights/FLIR_FROM_COCO_43.h5 &
-> da scrivere

## Per 29/10
---------------------------------------------------------------------------------------------------------------------------------

- [x] riaddestro su kaist termico con data augmentation: test su -> KAIST_DA e KAIST_DA_FLIR_TEST

- [x] riaddestro su FLIR termico con data augmentation:     
   - ADDESTRAMENTO TERMINATO, I RISULTATI SONO SIMILI A 9, PEGGIORA IN TRAINING

- [x] 9 riaddestro su FLIR termico con più classi E DA (auto) -> TESTO SU FLIR: FLIR_DA_CARS_1

  - retinanet-evaluate --score-threshold 0.30 --save-path ./FLIR_DA_CARS_1 --convert-model --gpu 0 csv ../FLIR/thermal_validation_KAIST_TEST_WCARS.csv ../FLIR/class_name_to_id_CARS.csv  ./FLIR_DA_CARS_1.h5 &



- [x] TESTO I PESI DI 9 SU KAIST: *FLIR_DA_CARS_1_ON_KAIST*

  - retinanet-evaluate --score-threshold 0.30 --save-path ./FLIR_DA_CARS_1_ON_KAIST --convert-model --gpu 0 csv ../KAIST_MPD/imageSets/csv_files_NO_PEOPLE/lwir/test-all-01.csv ../KAIST_MPD/class_name_to_ID_CARS.csv  ./FLIR_DA_CARS_1.h5


## Per 5/10
---------------------------------------------------------------------------------------------------------------------------------

- [x] annotare le auto manualmente su kaist
- [x] ri-addestrare: 
   - fatto fine tuning su kaist partendo dai pesi di FLIR
   - Nome run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2*

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST: 
  - 755 instances of class person with average precision: 0.4964
  - 15 instances of class cyclist with average precision: 0.0333
  - 1088 instances of class cars with average precision: 0.6767
  - mAP using the weighted average of precisions among classes: *0.5983*
  - mAP: 0.4022

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 10 (manual_annotation_10.h5): 
  - 755 instances of class person with average precision: 0.5050
  - 15 instances of class cyclist with average precision: 0.0444
  - 1088 instances of class cars with average precision: 0.6929
  - mAP using the weighted average of precisions among classes: *0.6113*
  - mAP: 0.4141 

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 9 (manual_annotation_09.h5): 
  - 755 instances of class person with average precision: 0.5018
  - 15 instances of class cyclist with average precision: 0.0667
  - 1088 instances of class cars with average precision: 0.6854
  - mAP using the weighted average of precisions among classes: *0.6058*
  - mAP: 0.4180 

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 8 (manual_annotation_08.h5): 
  - 755 instances of class person with average precision: 0.5137
  - 15 instances of class cyclist with average precision: 0.1000
  - 1088 instances of class cars with average precision: 0.6787
  - mAP using the weighted average of precisions among classes: *0.6070*
  - mAP: 0.4308 
  
- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 8 (manual_annotation_05.h5): 
  - 755 instances of class person with average precision: 0.5048
  - 15 instances of class cyclist with average precision: 0.0667
  - 1088 instances of class cars with average precision: 0.6870
  - mAP using the weighted average of precisions among classes: *0.6080*
  - mAP: 0.4195


- [x] test di pesi da run *FLIR FROM COCO [DA, CARS]* su KAIST annotato manualmente (nome cartella: *FLIR_DA_CARS_1_MANUAL_ANNOTATION*):
  - 755 instances of class person with average precision: 0.3730
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5716
  - mAP using the weighted average of precisions among classes: *0.4863*
  - mAP: 0.3149
  
## Test con Auto Augment
Prove effettuate con policy v0:
- [x] Addestramento partito dai migliori pesi del fine tuning di KAIST, ovvero manual_annotation_08.h5, epoca 11:
   - 755 instances of class person with average precision: 0.4982
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6792
   - mAP using the weighted average of precisions among classes: *0.6002*
   - mAP: 0.3925
- [x] Addestramento partito dai migliori pesi del fine tuning di KAIST, ovvero manual_annotation_08.h5, epoca 8:
   - 755 instances of class person with average precision: 0.4850
   - 15 instances of class cyclist with average precision: 0.0167
   - 1088 instances of class cars with average precision: 0.6942
   - mAP using the weighted average of precisions among classes: *0.6037*
   - mAP: 0.3986
- [x] Addestramento partito dai migliori pesi del fine tuning di KAIST, ovvero manual_annotation_08.h5, epoca 5:
   - 755 instances of class person with average precision: 0.4808
   - 15 instances of class cyclist with average precision: 0.0067
   - 1088 instances of class cars with average precision: 0.7195
   - mAP using the weighted average of precisions among classes: *0.6167*
   - mAP: 0.4023

Prove effettuate con policy v1:
- [x] epoca 05:
   - 755 instances of class person with average precision: 0.5157
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6805
   - mAP using the weighted average of precisions among classes: *0.6080*
   - mAP: 0.3987
- [x]  epoca 10:
   - 755 instances of class person with average precision: 0.5004
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6728
   - mAP using the weighted average of precisions among classes: **0.5973*
   - mAP: 0.3911
- [x]  epoca 15:
   - 755 instances of class person with average precision: 0.4896
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6605
   - mAP using the weighted average of precisions among classes: *0.5857*
   - mAP: 0.3834
- [x] epoca 20:
   - 755 instances of class person with average precision: 0.4876
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6703
   - mAP using the weighted average of precisions among classes: *0.5906*
   - mAP: 0.3859
- [x] epoca 30:
   - 755 instances of class person with average precision: 0.4975
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6693
   - mAP using the weighted average of precisions among classes: *0.5941*
   - mAP: 0.3889
- [x] epoca 33 (ultima):
   - 755 instances of class person with average precision: 0.3873
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.5603
   - mAP using the weighted average of precisions among classes: *0.4855*
   - mAP: 0.3159


Prove effettuate con policy v2:
- [x] epoca 05:
   - 755 instances of class person with average precision: *0.5241* (MIGLIORE SULLE PERSONE!)
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6506
   - mAP using the weighted average of precisions among classes: *0.5940*
   - mAP: 0.3916
- [x]  epoca 10:
   - 755 instances of class person with average precision: 0.5032
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6564
   - mAP using the weighted average of precisions among classes: *0.5888*
   - mAP: 0.3865
- [x]  epoca 15:
   - 755 instances of class person with average precision: 0.5007
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6505
   - mAP using the weighted average of precisions among classes: *0.5844*
   - mAP: 0.3838
- [x] epoca 20:
   - 755 instances of class person with average precision: 0.4959
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6254
   - mAP using the weighted average of precisions among classes: *0.5677*
   - mAP: 0.3738
- [x] epoca 30:
   - 755 instances of class person with average precision: 0.5064
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6264  
   - mAP using the weighted average of precisions among classes: *0.5726*
   - mAP: 0.3776
- [x] epoca 37 (ultima):
   - 755 instances of class person with average precision: 0.5064
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6265
   - mAP using the weighted average of precisions among classes: *0.5726*
   - mAP: 0.3776

## AutoAugment, policy v2 da FLIR a KAIST:
- [x] test FLIR su KAIST, epoca 5:
  - 755 instances of class person with average precision: 0.3342
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.0000
  - mAP using the weighted average of precisions among classes: 0.1358
  - mAP: 0.1114
- [x] test FLIR su KAIST, epoca 20:
  - 755 instances of class person with average precision: 0.2516
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.0000
  - mAP using the weighted average of precisions among classes: 0.1022
  - mAP: 0.0839
- [x] test FLIR su KAIST, epoca 40:
  - 755 instances of class person with average precision: 0.3619
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.0000
  - mAP using the weighted average of precisions among classes: 0.1471
  - mAP: 0.1206
- [x] test FLIR su KAIST, epoca 50:
  - 755 instances of class person with average precision: 0.3270
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.0000
  - mAP using the weighted average of precisions among classes: 0.1329
  - mAP: 0.1090
- [x] test FLIR su KAIST, epoca 63:
  - 755 instances of class person with average precision: 0.3531
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.0000
  - mAP using the weighted average of precisions among classes: 0.1435
  - mAP: 0.1177


----------------------------------------------------------
- [x] test epoca 5, treshold 0.3: 
  - 755 instances of class person with average precision: 0.5061
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5250
  - mAP using the weighted average of precisions among classes: 0.5131
  - mAP: 0.3437
- [x] test epoca 5, treshold 0.5: 
  - 755 instances of class person with average precision: 0.4494
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.4691
  - mAP using the weighted average of precisions among classes: 0.4573
  - mAP: 0.3062
- [x] test epoca 10, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4899
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5395
  - mAP using the weighted average of precisions among classes: 0.5150
  - mAP: 0.3431
- [x] test epoca 15, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4913
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5360
  - mAP using the weighted average of precisions among classes: 0.5135
  - mAP: 0.3424
- [x] test epoca 20, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4959
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5383
  - mAP using the weighted average of precisions among classes: 0.5167
  - mAP: 0.3447
- [x] test epoca 25, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4822
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5462
  - mAP using the weighted average of precisions among classes: 0.5158
  - mAP: 0.3428
- [x] test epoca 30, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4847
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5427
  - mAP using the weighted average of precisions among classes: 0.5148
  - mAP: 0.3425
- [x] test epoca 35, treshold 0.3: 
  - 755 instances of class person with average precision: 0.5035
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5454
  - mAP using the weighted average of precisions among classes: 0.5239
  - mAP: 0.3496
- [x] test epoca 40, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4859
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5387
  - mAP using the weighted average of precisions among classes: 0.5129
  - mAP: 0.3415
- [x] test epoca 44, treshold 0.3: 
  - 755 instances of class person with average precision: 0.4911
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5399
  - mAP using the weighted average of precisions among classes: 0.5157
  - mAP: 0.3437


## Test dei pesi migliori sulla classe PERSON (POLICY V2 EPOCA 05)
- [x] Policy v2 epoca 05 (threshold 0.3):
   - 647 instances of class person with average precision: 0.5084
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6510
   - mAP using the weighted average of precisions among classes: 0.5927
   - mAP: 0.3865
- [x] Policy v2 epoca 05 (threshold 0.5):
   - 647 instances of class person with average precision: 0.4754
   - 15 instances of class cyclist with average precision: 0.0000
   - 1088 instances of class cars with average precision: 0.6048
   - mAP using the weighted average of precisions among classes: 0.5518
   - mAP: 0.3601

- [x] manual_annotations08.h5(threshold 0.3):
   - 647 instances of class person with average precision: 0.4795
   - 15 instances of class cyclist with average precision: 0.1000
   - 1088 instances of class cars with average precision: 0.6626663816638166381
   - mAP using the weighted average of precisions among classes: 0.5901
   - mAP: 0.4140


### Test dopo ottimizzazione di randaugment partendo da manual_annotations_08.h5
- N = 3
- M = 26
## test fatto su set manuale senza people:
- [x] Miglior risultato ottenuto all'epoca 2 (threshold 0.3):
   - 647 instances of class person with average precision: 0.476908
   - 15 instances of class cyclist with average precision: 0.033333
   - 1088 instances of class cars with average precision: 0.664653
   - mAP using the weighted average of precisions among classes: 0.589830
   - mAP: 0.391631
## test fatto su set manuale con people:
- [x] Miglior risultato ottenuto all'epoca 2 (threshold 0.3):
   - 755 instances of class person with average precision: 0.476908
   - 15 instances of class cyclist with average precision: 0.033333
   - 1088 instances of class cars with average precision: 0.663814
   - mAP using the weighted average of precisions among classes: 0.583137
   - mAP: 0.3916
