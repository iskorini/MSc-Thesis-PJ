### CHANGELOG PRELIMINARE DEL FILE autoaugment_utils_tf2.py
--------------------------------------------------------
Per farlo funzionare con Tensorflow 2 sono state apportate le seguenti modifiche:
- Passaggio sullo script tf2_upgrade
- Rimozione di `from tensorflow.contrib import image as contrib_image` e passaggio a `from tensorflow_addons import image as contrib_image`
- Rimozione di `from tensorflow.contrib import training as contrib_training` e passaggio di hparams a dizionario con conseguente modifica nelle parti dove è utilizzato
  

### TEST E FIX DELLE FUNZIONI NEL FILE autoaugment_utils_tf2.py
--------------------------------------------------------
- `TranslateX_BBox`: fatto bugfix su booleano shift_horizontal. Bounding Box non precise, vedere `_shift_bbox()`.
- `TranslateY_BBox`: uguale a `TranslateX_BBox`.
- `ShearX_BBox`: cambiato valore del booleano `shear_horizontal`. Bounding box errate del 60% o del 25% a seconda di X o Y
- `ShearY_BBox`: uguale a precedente
- `Cutout`: funziona con riserva
- `BBox_Cutout`: da quello che ho capito dovrebbe funzionare come `Cutout` applicato all'interno di una bounding box scelta casualmente. A volte lo applica anche al di fuori delle bounding box, devo verificare se questo comportamento è corretto.
- `Rotate_BBox`: corretta la variabile `rotation_matrix`.
- `Rotate_Only_BBoxes`: 
- `ShearX_Only_BBoxes`:
- `ShearY_Only_BBoxes`:
- `TranslateX_Only_BBoxes`:
- `TranslateY_Only_BBoxes`:
- `Flip_Only_BBoxes`:
- `Solarize_Only_BBoxes`:
- `Equalize_Only_BBoxes`:
- `Cutout_Only_BBoxes`:
- `Equalize`:
- `AutoContrast`:
- `Equalize`: sembra funzionare
- `Posterize`:
- `Solarize`:
- `SolarizeAdd`:
- `Color`:
- `Contrast`:
- `Brightness`:
- `Sharpness`:

