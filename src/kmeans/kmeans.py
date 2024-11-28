import matplotlib.pyplot as plt

def read_kmeans_result(filename):
    centroids = []
    datapoints = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        reading_centroids = False
        reading_datapoints = False

        for line in lines:
            if "Final Centroids" in line:
                reading_centroids = True
                reading_datapoints = False
                continue
            elif "Data Points" in line:
                reading_centroids = False
                reading_datapoints = True
                continue

            if reading_centroids:
                parts = line.strip().split(":")
                if len(parts) < 2:  # parts 리스트의 길이를 확인
                    continue
                coordinates = list(map(float, parts[1].strip().split()))
                centroids.append(coordinates)
            elif reading_datapoints:
                parts = line.strip().split(":")
                if len(parts) < 2:  # parts 리스트의 길이를 확인
                    continue
                data_part = parts[1].split("->")[0].strip()  # "->" 이전의 좌표 부분만 가져오기
                coordinates = list(map(float, data_part.split()))
                cluster = int(parts[1].split()[-1])  # 마지막 클러스터 ID 추출
                datapoints.append((coordinates, cluster))

    return centroids, datapoints

def plot_kmeans(centroids, datapoints, output_filename="kmeans_result.png"):
    plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 최대 7개의 클러스터만 다른 색으로 표시

    # 데이터 포인트 시각화
    for point, cluster in datapoints:
        plt.scatter(point[0], point[1], color=colors[cluster % len(colors)], alpha=0.5)

    # 클러스터 중심 시각화
    for idx, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color=colors[idx % len(colors)], marker='X', s=200, edgecolors='black', label=f'Centroid {idx + 1}')

    plt.title('K-means Clustering Result')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # 이미지 파일로 저장
    plt.savefig(output_filename)
    plt.close()  # 리소스를 절약하기 위해 plt 닫기

if __name__ == "__main__":
    centroids, datapoints = read_kmeans_result("kmeans_result.txt")
    plot_kmeans(centroids, datapoints, "kmeans_result.png")
