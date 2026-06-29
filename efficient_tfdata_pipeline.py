"""
efficient_tfdata_pipeline.py
Example: Efficient Data Pipelines with tf.data
"""
import tensorflow as tf

CHATGPT_PROMPT = """
Generate a TensorFlow tf.data pipeline that loads images, preprocesses them,
uses parallel mapping, caching, batching, and prefetching.
"""

IMG_HEIGHT=224
IMG_WIDTH=224
BATCH_SIZE=32

image_paths=["sample1.jpg","sample2.jpg","sample3.jpg"]
labels=[0,1,0]

def preprocess(path,label):
    image=tf.io.read_file(path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,(IMG_HEIGHT,IMG_WIDTH))
    image=image/255.0
    return image,label

dataset=tf.data.Dataset.from_tensor_slices((image_paths,labels))
dataset=(dataset
         .shuffle(len(image_paths))
         .map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
         .cache()
         .batch(BATCH_SIZE)
         .prefetch(tf.data.AUTOTUNE))

print(dataset)

print("\nSuggested ChatGPT prompts:")
print("- Add data augmentation")
print("- Read TFRecord files")
print("- Optimize for multi-GPU")
