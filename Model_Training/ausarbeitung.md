# VibeluX
Tim Schacht, Quirin Barth

## Idea/Scope

The goal is to train two convolutional neural networks, both of which will classify their given input into one of seven emotions: 
anger, disgust, fear, happy, neutral, sad, and surprise.
One of the models is trained to recognize faces, while the other is trained to recognize these emotions in music. Together the predictions of both models can be used to then recommend music to a user, based on their current emotional state. Other applications include some hidden internal feature that some music app might use, to subtly adjust their shuffle algorithm in a way that better fits the mood of the user.

To facilitate training, a dataset of faces was obtained from kaggle: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

The music used to train the other model was acquired from a royalty-free use sound website: https://pixabay.com/music/


## Theorie

### What are CNNs?
Make thing into chunk, look at chunk, make smaller chunk, repeat, it can look at things by breaking them down!
Mathematically? You're "folding" teh data across itself according to some rules, resulting in a smaller output, that is denser in information eg. (8 x 8 x 1) -> (4 x 4 x 4)

Simillarly to multilevel perceptrons, CNNs are based loosely on how human perception works, just in a more direct way in that it applies "filters" to an image, which are effectively a sliding window that checks for certain patterns, and outputs how present that pattern is. The resulting map of patterns can then be fed into the next CNN layer to recognizer larger continuous patterns! This means that by looking at the filters one might imagine being able to see what the CNN is thinking! This is sadly not actually the case, as there are many many many possible stable and functional local minima in the loss-function's space, resulting in most CNN filters being just as humanly incomprehensible as any other neural network.

### Why maxpooling?
Squish the data, make it squished, now less data with hopefully still relevant information -> Less computation, yay :D
[TODO](Test is the music classifier might work marginally better without the pooling layer... unlikely though)

### Why Dropout?
Dropout randomly "disables" neurons or equivalents in a network during training, and rescales outputs to still be properly backpropagationable. This is a regularization method used to reduce overfitting, and in general force the model to learn more relevant features spread across more of it's neurons, which helps the network actually use more neurons instead of just ignoring certain neurons later in it's training (see relu getting stuck at 0 and "dying")

### Face Classification dataset
48x48 greyscale faces!

### Music Classification dataset
Songs that were tagged with their emotion or a synonym for that emotion were picked, all cropped/padded to be 1:30 long, and then converted into Spectrograms.

### What are Spectrograms?
Music is frequencies? Fourier transform to get frequencies! plot frequencies (y) over time (x) with their volume/amplitude as a color (greyscale) and you get a spectrogram fully describing the audio!!! These can be converted back into audio even! But for the purposes of this Project, it means we can turn music into images that fully describe the music. By clipping or buffering the music to always be 1:30 long, we can ensure the images always have the same format, letting us use them as a CNN imput without much difficulty.

Why though? Because previously research has shown that CNN's don't really care if the image is human readable, they just care if the image contains details that can lead to conclusions about the classification. As such, it has been proven that one can train a CNN on music spectrograms, and achieve good results! [TODO](QUIRIN GO FIND SOME SOURCES)

### Model Architecture
3x (CNN-Maxpool-Dropout)
2x (Dense with ever decreasing neuron count)
1x Dense layer with 7 neurons and softmax activation function)

### Emotional Embedding search
Both models produce an emotional "vector", classifying the face or music into an emotion. Music has many facets, and so instead of just detecting the primary emotion and picking songs that were classified with a fitting emotion, we instead opted to use cosine-similarity to pick the song(s) that have the closest match to all 7 emotions detected in the face.

## Result
Face recognition? Works! A simple network was thrown together in tensorflow, and trained over less than 10 minutes, and has yielded acceptable results! [TODO](I realize now, I should have written down it's accuracy :P Oops)

Music recognition? Training approaches a loss of 1.94, which it then never improves on. This is likely due to a lack of training data, as there are currently only 80 songs per emotion (total of 560 songs), and a song's spectrogram is a rather large image (~1k), requiring a larger model than the face recognition's mere 48x48. Music is also much much more varied in expression than a human face is, meaning there are many different ways to make a "happy" song. This iresults in the model seemingly having a very difficult time picking out relevant details from the tiny amount of training data.

Embedding search via cosine-similarity? Works in testing, but the music classifier doesn't work properly, so we cannot currently generate **real** embeddings for songs.

