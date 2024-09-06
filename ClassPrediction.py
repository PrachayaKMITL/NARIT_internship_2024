import cv2
import numpy as np
from preprocessing import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import base64

class prediction:
    def __init__(self):
        pass
    def RBsingle(self, input):
        final = []
        value= []
        chan_b = []
        chan_r = []
        e = 1e-11
        for i in input:
            R, _, B = cv2.split(i)
            B = B + e
            chan_b.append(np.mean(B))
            chan_r.append(np.mean(R))
            intensity = np.mean(B)
            ratio = (R/B)*intensity/9
            ratio = cv2.convertScaleAbs(ratio)
            final_mask = cv2.threshold(ratio, intensity/20, 255, cv2.THRESH_BINARY)[1]
            masked = cv2.bitwise_and(i, i, mask=final_mask)
            masked_gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
            final.append(masked_gray)
            value.append(intensity)
        chan_r = np.array(chan_r).reshape(-1, 1)
        chan_b = np.array(chan_b).reshape(-1, 1)
        RB = np.concatenate((chan_r, chan_b), axis=1)
        return final, value, RB.T

    def CloudRatio(self, image, mask):
        image = np.array(image)
        mask = np.array(mask)
        area = cv2.countNonZero(image)
        mask = cv2.countNonZero(mask)
        return area / mask * 100, np.std(image)
    def octas(self, cloud_percentage,std):
        oktas = round((cloud_percentage / 100) * 8)
        if oktas == 8 and std < 68:
            return "Thin cloud(9 octas)"
        return f"{oktas} oktas"
    def classify_sky(self, cloud_percentage, std):
        oktas = round((cloud_percentage / 100) * 8)
        
        # Classifications based on oktas value
        classifications = {
            0: "Clear sky",
            1: "Fewer clouds",
            2: "Few clouds",
            3: "Scatter",
            4: "Mostly Scatter",
            5: "Partly Broken",
            6: "Mostly Broken",
            7: "Broken",
            8: "Overcast"
        }
        
        # If oktas is "Overcast" (i.e., 8) and std is lower than 60, classify as "High cloud"
        if oktas == 8 and std < 70:
            classification = "Thin cloud"
        else:
            classification = classifications.get(oktas, 'Invalid cloud percentage')

        return f"{classification} ({oktas} okta{'s' if oktas != 1 else ''})"
    def sky_status(self,cloud_percent):
        if cloud_percent < 18:
            return "Clear"
        elif cloud_percent < 37:
            return "Mostly clear"
        elif cloud_percent < 55:
            return "Partly cloudy"
        elif cloud_percent < 64:
            return "Mostly Cloudy"
        elif cloud_percent < 87:
            return "Cloudy"
        else:
            return "Overcast"
    def total_prediction(self, image_path, mask_path, crop_size=570, properties=None,sunrise=None,sunset=None,kmeans=None, miniBatchesKmeans=None):
        if properties is None:
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = preprocessData().crop_center(mask, crop_size=crop_size)
        images, filename = preprocessData().load_single_image(image_path, mask=mask, crop_size=crop_size, apply_crop_sun=True)
        final,intensity,_ = thresholding().RBratiosingle(input=images[0],filename=filename[0],sunrise=sunrise,sunset=sunset)
        glcm = [preprocessData().computeGlcmsingle(image=final, distance=[3], angle=[45])]
        test = preprocessData().getDataframe(property=properties, gray_level=glcm, index=filename,intensity=None, statistical=None)
        x = test.drop(columns=['correlation'])
        predict_1 = kmeans.predict(x)
        predict_2 = miniBatchesKmeans.predict(x)
        cloud_ratio,std = self.CloudRatio(image=final,mask=mask)
        sky_status = self.octas(cloud_ratio,std)
        return [predict_1,predict_2,cloud_ratio,sky_status,final,intensity]
    def weighted_prediction(self,weight:None,predicted_result:list,intensity,cloud_percent:float,sky_status=None):
        if weight is None:
            weight = [0.5, 0.7, 0.7, 0.4]
        risk_factor =  ((min(predicted_result[0],predicted_result[1])*2)+cloud_percent+intensity)/(108+intensity)*100
        return risk_factor
class visualizer:
    def __init__(self):
        pass
    def match_label(self, df, number):
        return df[df['Label_GMM'] == number]
    def copy_matching_files(self, df, source_folder, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        indices = [str(index) for index in df.index.tolist()]
        all_files = os.listdir(source_folder)
        matching_files = []
        for file in all_files:
            name = os.path.splitext(os.path.basename(file))[0]
            if name in indices:
                matching_files.append(name + ".png")
        for filename in matching_files:
            src_file = os.path.join(source_folder, filename)
            dst_file = os.path.join(destination_folder, filename)
            shutil.copy(src_file, dst_file)
    def image_to_base64(self,image):
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    def image_html(self,image_base64,size:list):
        return f'<img src="data:image/png;base64,{image_base64}" width="{size[0]}" height="{size[1]}"/>'
    def progress_bar(self,progress, total, bar_length=50):
        percent = progress / total
        arrow = '=' * int((percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        percent = percent*100
        print(f'\rProgress: [{arrow}{spaces}] {percent:.3f}%', end='')

class Evaluation:
    def __init__(self):
        pass
    def silhouette(self,data, n):
        k=range(2,n)
        s = []
        for n_clusters in k:
            clusters = KMeans(n_clusters=n_clusters,tol=1e-4, random_state=42, init='k-means++' ,n_init='auto', max_iter=500, algorithm='lloyd')
            clusters.fit(data)
            labels = clusters.labels_
            centroids = clusters.cluster_centers_
            s.append(silhouette_score(data, labels, metric='euclidean'))
            print("Silhouette Coefficient for k == %s: %s" % (
            n_clusters, round(silhouette_score(data, clusters.labels_), 4)))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k,s,'b*-')
        ax.plot(k[np.argmax(s)], s[np.argmax(s)], marker='o', markersize=12,markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        plt.ylabel("Silouette Score")
        plt.xlabel("Number of clusters")
        plt.title("Silouette for KMeans clustering")
        plt.show()
    def silhouette_GMM(self,data, max_n_clusters):
        k = range(2, max_n_clusters)  # Range of cluster numbers to try
        s = []

        for n_clusters in k:
            # Fit Gaussian Mixture Model with the specified number of components
            gmm = GaussianMixture(n_components=n_clusters, max_iter=200, random_state=42, n_init=20)
            gmm.fit(data)
            
            # Predict cluster labels
            labels = gmm.predict(data)
            
            # Calculate silhouette score
            score = silhouette_score(data, labels, metric='euclidean')
            s.append(score)
            
            # Print silhouette score for the current number of clusters
            print(f"Silhouette Coefficient for n_clusters == {n_clusters}: {score:.4f}")

        # Plot silhouette scores for different numbers of clusters
        fig, ax = plt.subplots()
        ax.plot(k, s, 'b*-')
        ax.plot(k[s.index(max(s))], max(s), marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        plt.ylabel("Silhouette Score")
        plt.xlabel("Number of clusters")
        plt.title("Silhouette Score for GMM clustering")
    def evaluate_gmm_bic(data, max_n_clusters):
        n_clusters_range = range(1, max_n_clusters + 1)
        bics = []
        for n_clusters in n_clusters_range:
            gmm = GaussianMixture(n_components=n_clusters, max_iter=200, random_state=42, n_init=20)
            gmm.fit(data)
            bics.append(gmm.bic(data))
        best_n_clusters = n_clusters_range[np.argmin(bics)]
        plt.figure(figsize=(8, 5))
        plt.plot(n_clusters_range, bics, 'bo-', label='BIC')
        plt.axvline(x=best_n_clusters, color='r', linestyle='--', label=f'Best number of clusters: {best_n_clusters}')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC')
        plt.title('BIC Scores for GMM')
        plt.legend()
        plt.show()
        plt.show()