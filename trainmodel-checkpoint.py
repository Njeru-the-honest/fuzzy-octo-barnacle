{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3b8af21a-7f5e-4345-a66b-48975c1f0712",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'keras_preprocessing'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mutils\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m to_categorical\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mkeras_preprocessing\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mimage\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_img\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Sequential\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlayers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Dense, Conv2D, Dropout, Flatten, MaxPooling2D\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'keras_preprocessing'"
     ]
    }
   ],
   "source": [
    "from keras.utils import to_categorical\n",
    "from keras_preprocessing.image import load_img\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd8482ee-52e1-419f-8ef3-c0a7298ce3cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "TRAIN_DIR = 'images/train'\n",
    "TEST_DIR = 'images/test'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd52699b-5ea0-4895-bf7f-4593bfd187ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "def createdataframe(dir):\n",
    "    image_paths = []\n",
    "    labels = []\n",
    "    for label in os.listdir(dir):\n",
    "        for imagename in os.listdir(os.path.join(dir,label)):\n",
    "            image_paths.append(os.path.join(dir,label,imagename))\n",
    "            labels.append(label)\n",
    "        print(label, \"completed\")\n",
    "    return image_paths,labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "412a5dc6-a428-4bf0-be86-1b56a2a52e68",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.DataFrame()\n",
    "train['image'], train['label'] = createdataframe(TRAIN_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4d44f9f-49ab-4eee-9b90-cb4d89b93d95",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "893eb15b-95c9-4d69-8f29-5d40e44fe6be",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.DataFrame()\n",
    "test['image'], test['label'] = createdataframe(TEST_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a41d544e-701c-42eb-a43a-566ee948b20b",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(test)\n",
    "print(test['image'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e4ffc57-d855-4b69-869b-f37dde7e61c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tqdm.notebook import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54a9ea53-380c-4765-bedc-1d6051dad906",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_features(images):\n",
    "    features = []\n",
    "    for image in tqdm(images):\n",
    "        img = load_img(image,grayscale =  True )\n",
    "        img = np.array(img)\n",
    "        features.append(img)\n",
    "    features = np.array(features)\n",
    "    features = features.reshape(len(features),48,48,1)\n",
    "    return features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9460ff1c-d807-4e22-80d1-a8d065c5986b",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_features = extract_features(train['image']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4a36f72-08d6-4a39-8501-d54292ed3f7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_features = extract_features(test['image'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97034ea1-59ae-4c13-9b26-64b439d1e488",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = train_features/255.0\n",
    "x_test = test_features/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14c1e525-f35d-45b8-ab55-94017dd15f7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87952975-23f5-4280-bd62-17cdb5ad5fed",
   "metadata": {},
   "outputs": [],
   "source": [
    "le = LabelEncoder()\n",
    "le.fit(train['label'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "35114309",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = le.transform(train['label'])\n",
    "y_test = le.transform(test['label'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffef34a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = to_categorical(y_train, num_classes=7)\n",
    "y_test = to_categorical(y_test, num_classes=7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c50d98d2-a6c0-4843-bba6-57eece25f1dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "# convolutional layers\n",
    "model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.4))\n",
    "\n",
    "model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.4))\n",
    "\n",
    "model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.4))\n",
    "\n",
    "model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.4))\n",
    "\n",
    "model.add(Flatten())\n",
    "# fully connected layers\n",
    "model.add(Dense(512, activation='relu'))\n",
    "model.add(Dropout(0.4))\n",
    "model.add(Dense(256, activation='relu'))\n",
    "model.add(Dropout(0.3))\n",
    "# output layer\n",
    "model.add(Dense(7, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "048e5b1e-fa64-44a2-88b4-0b37f75288a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "467ae4fd-40a0-445b-ba2e-5be836928048",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test)) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d476a888-ab7b-43d3-ba9f-d1e6ae455401",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_json = model.to_json()\n",
    "with open(\"emotiondetector.json\",'w') as json_file:\n",
    "    json_file.write(model_json)\n",
    "model.save(\"emotiondetector.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d87a1581-7ee6-4ef7-82f9-8fa6ba25168d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import model_from_json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a916de6-178b-4d1d-b90d-07d2c322a4cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "json_file = open(\"facialemotionmodel.json\", \"r\")\n",
    "model_json = json_file.read()\n",
    "json_file.close()\n",
    "model = model_from_json(model_json)\n",
    "model.load_weights(\"facialemotionmodel.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e6b7672-961a-4e69-a964-b1be0e4831c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']\n",
    "def ef(image):\n",
    "    img = load_img(image,grayscale =  True )\n",
    "    feature = np.array(img)\n",
    "    feature = feature.reshape(1,48,48,1)\n",
    "    return feature/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63e92a15-64f0-4975-b29d-0ba87005195f",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/sad/42.jpg'\n",
    "print(\"original image is of sad\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e969c6c3-35d7-4060-bf70-37e37acf92a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "745f150b-03d5-46ee-b8a2-858bcd145926",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/sad/42.jpg'\n",
    "print(\"original image is of sad\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)\n",
    "plt.imshow(img.reshape(48,48),cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "255f1f6d-0e0a-4f73-8b7f-e623ef4d881b",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/fear/2.jpg'\n",
    "print(\"original image is of fear\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)\n",
    "plt.imshow(img.reshape(48,48),cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9953139b-a595-47f6-a56c-042ecea5ef79",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/disgust/299.jpg'\n",
    "print(\"original image is of disgust\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)\n",
    "plt.imshow(img.reshape(48,48),cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6dfef67-2a46-4618-970d-de04713fc09b",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/happy/7.jpg'\n",
    "print(\"original image is of happy\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)\n",
    "plt.imshow(img.reshape(48,48),cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3b54d0b-9631-46b1-84ce-f41247f8a7c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = 'images/train/surprise/15.jpg'\n",
    "print(\"original image is of surprise\")\n",
    "img = ef(image)\n",
    "pred = model.predict(img)\n",
    "pred_label = label[pred.argmax()]\n",
    "print(\"model prediction is \",pred_label)\n",
    "plt.imshow(img.reshape(48,48),cmap='gray')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
