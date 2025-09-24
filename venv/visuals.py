import matplotlib as plt
import numpy as np
from model1st_version import model, train_dataset, validation_dataset, history 

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

val_loss, val_acc = model.evaluate(validation_dataset, verbose=2)

print("\nAcurácia de validação final:", val_acc)