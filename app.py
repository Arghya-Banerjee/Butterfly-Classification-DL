import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('./butterfly_classification_inception.h5')

def classify_butterfly(img):
    # Make prediction
#     img = image.load_img(file_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)


    # Decoding prediction

    labels = {0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88', 4: 'APPOLLO', 5: 'ARCIGERA FLOWER MOTH', 6: 'ATALA', 7: 'ATLAS MOTH', 8: 'BANDED ORANGE HELICONIAN', 9: 'BANDED PEACOCK', 10: 'BANDED TIGER MOTH', 11: 'BECKERS WHITE', 12: 'BIRD CHERRY ERMINE MOTH', 13: 'BLACK HAIRSTREAK', 14: 'BLUE MORPHO', 15: 'BLUE SPOTTED CROW', 16: 'BROOKES BIRDWING', 17: 'BROWN ARGUS', 18: 'BROWN SIPROETA', 19: 'CABBAGE WHITE', 20: 'CAIRNS BIRDWING', 21: 'CHALK HILL BLUE', 22: 'CHECQUERED SKIPPER', 23: 'CHESTNUT', 24: 'CINNABAR MOTH', 25: 'CLEARWING MOTH', 26: 'CLEOPATRA', 27: 'CLODIUS PARNASSIAN', 28: 'CLOUDED SULPHUR', 29: 'COMET MOTH', 30: 'COMMON BANDED AWL', 31: 'COMMON WOOD-NYMPH', 32: 'COPPER TAIL', 33: 'CRECENT', 34: 'CRIMSON PATCH', 35: 'DANAID EGGFLY', 36: 'EASTERN COMA', 37: 'EASTERN DAPPLE WHITE', 38: 'EASTERN PINE ELFIN', 39: 'ELBOWED PIERROT', 40: 'EMPEROR GUM MOTH', 41: 'GARDEN TIGER MOTH', 42: 'GIANT LEOPARD MOTH', 43: 'GLITTERING SAPPHIRE', 44: 'GOLD BANDED', 45: 'GREAT EGGFLY', 46: 'GREAT JAY', 47: 'GREEN CELLED CATTLEHEART', 48: 'GREEN HAIRSTREAK', 49: 'GREY HAIRSTREAK', 50: 'HERCULES MOTH', 51: 'HUMMING BIRD HAWK MOTH', 52: 'INDRA SWALLOW', 53: 'IO MOTH', 54: 'Iphiclus sister', 55: 'JULIA', 56: 'LARGE MARBLE', 57: 'LUNA MOTH', 58: 'MADAGASCAN SUNSET MOTH', 59: 'MALACHITE', 60: 'MANGROVE SKIPPER', 61: 'MESTRA', 62: 'METALMARK', 63: 'MILBERTS TORTOISESHELL', 64: 'MONARCH', 65: 'MOURNING CLOAK', 66: 'OLEANDER HAWK MOTH', 67: 'ORANGE OAKLEAF', 68: 'ORANGE TIP', 69: 'ORCHARD SWALLOW', 70: 'PAINTED LADY', 71: 'PAPER KITE', 72: 'PEACOCK', 73: 'PINE WHITE', 74: 'PIPEVINE SWALLOW', 75: 'POLYPHEMUS MOTH', 76: 'POPINJAY', 77: 'PURPLE HAIRSTREAK', 78: 'PURPLISH COPPER', 79: 'QUESTION MARK', 80: 'RED ADMIRAL', 81: 'RED CRACKER', 82: 'RED POSTMAN', 83: 'RED SPOTTED PURPLE', 84: 'ROSY MAPLE MOTH', 85: 'SCARCE SWALLOW', 86: 'SILVER SPOT SKIPPER', 87: 'SIXSPOT BURNET MOTH', 88: 'SLEEPY ORANGE', 89: 'SOOTYWING', 90: 'SOUTHERN DOGFACE', 91: 'STRAITED QUEEN', 92: 'TROPICAL LEAFWING', 93: 'TWO BARRED FLASHER', 94: 'ULYSES', 95: 'VICEROY', 96: 'WHITE LINED SPHINX MOTH', 97: 'WOOD SATYR', 98: 'YELLOW SWALLOW TAIL', 99: 'ZEBRA LONG WING'}

    result = labels[preds[0]]
    return result

img = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=3)

iface = gr.Interface(fn=classify_butterfly, inputs=img, outputs=label, interpretation='default')

iface.launch(inline=False)