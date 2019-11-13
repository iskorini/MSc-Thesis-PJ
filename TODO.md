## Step iniziale
---------------------------------------------------------------------------------------------------------------------------------

- [x] 1 ADDESTRAMENTO PARTITO DA COCO SU VISIBILE IN KAIST_MPD: annotated_test_visible

- [x] 2 ADDESTRAMENTO PARTITO DAI PESI DI 1 SU KAIST_MPD TERMICO: annotated_test_LWIR



- [x] TEST DEI PESI DI 2 SU FLIR TERMICO: annotated_test_FLIR


- [x] ADDESTRAMENTO PARTITO DAI PESI DI 2 SU FLIR TERMICO: ANNOTATED_FLIR_TRAINED


- [x] 5 ADDESTRAMENTO PARTITO DAI PESI DI COCO SU FLIR TERMICO: FLIR_FROM_COCO


- [x] TEST DEI PESI DI 5 SU KAIST TERMICO: KAIST_FROM_FLIR
nohup retinanet-evaluate --score-threshold 0.30 --save-path ./KAIST_FROM_FLIR  --convert-model --gpu 0 csv KAIST_MPD/imageSets/csv_files_NO_PEOPLE/lwir/test-all-01.csv KAIST_MPD/class_name_to_id_NO_PEOPLE.csv  weights/FLIR_FROM_COCO_43.h5 &

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
- [ ] normalizzare immagini di giorno e notte su kaist (non posso farlo su flir): creato script per calcolare la media, ma riguardare e chiedere per dubbi
- [x] ri-addestrare: 
   - fatto fine tuning su kaist partendo dai pesi di FLIR
   - Nome run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2*
- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST: 
  - 755 instances of class person with average precision: 0.4964
  - 15 instances of class cyclist with average precision: 0.0333
  - 1088 instances of class cars with average precision: 0.6767
  - mAP using the weighted average of precisions among classes: 0.5983
  - mAP: 0.4022

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 10 (manual_annotation_10.h5): 
  - 755 instances of class person with average precision: 0.5050
  - 15 instances of class cyclist with average precision: 0.0444
  - 1088 instances of class cars with average precision: 0.6929
  - mAP using the weighted average of precisions among classes: 0.6113
  - mAP: 0.4141 

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 9 (manual_annotation_09.h5): 
  - 755 instances of class person with average precision: 0.5018
  - 15 instances of class cyclist with average precision: 0.0667
  - 1088 instances of class cars with average precision: 0.6854
  - mAP using the weighted average of precisions among classes: 0.6058
  - mAP: 0.4180 

- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 8 (manual_annotation_08.h5): 
  - 755 instances of class person with average precision: 0.5137
  - 15 instances of class cyclist with average precision: 0.1000
  - 1088 instances of class cars with average precision: 0.6787
  - mAP using the weighted average of precisions among classes: 0.6070
  - mAP: 0.4308 
  
- [x] test di pesi da run *FINE TUNING: FLIR [CARS] -> KAIST_MPD 2* su set annotato manualmente in KAIST, a differenza di prima provo qualche epoca precedente,  EPOCA 8 (manual_annotation_05.h5): 
  - 755 instances of class person with average precision: 0.5048
  - 15 instances of class cyclist with average precision: 0.0667
  - 1088 instances of class cars with average precision: 0.6870
  - mAP using the weighted average of precisions among classes: 0.6080
  - mAP: 0.4195


- [x] test di pesi da run *FLIR FROM COCO [DA, CARS]* su KAIST annotato manualmente (nome cartella: *FLIR_DA_CARS_1_MANUAL_ANNOTATION*):
  - 755 instances of class person with average precision: 0.3730
  - 15 instances of class cyclist with average precision: 0.0000
  - 1088 instances of class cars with average precision: 0.5716
  - mAP using the weighted average of precisions among classes: 0.4863
  - mAP: 0.3149
  
