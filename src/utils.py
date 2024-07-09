import matplotlib.pyplot as plt

def visualize_pairs(dataset, num_pairs=5):
    plt.figure(figsize=(10, 5 * num_pairs))
    
    # Unbatch if the dataset is batched
    if isinstance(next(iter(dataset)), tuple):
        dataset = dataset.unbatch()
    
    for i, ((img1, img2), label) in enumerate(dataset.take(num_pairs)):
        # Convert the images from Tensor to NumPy array for visualization
        img1 = img1.numpy()
        img2 = img2.numpy()
        label = label.numpy()
        
        # Plot the first image in the pair
        ax = plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(img1.astype("uint8"))
        plt.title(f"Image 1 - Pair {i+1}")
        plt.axis("off")
        
        # Plot the second image in the pair
        ax = plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(img2.astype("uint8"))
        plt.title(f"Image 2 - Pair {i+1}\nLabel: {'Same Class' if label == 1 else 'Different Class'}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
