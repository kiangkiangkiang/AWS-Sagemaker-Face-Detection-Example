import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="ap-northeast-1")
endpoint_name = "luka-face-detection-test-endpoint1"


def visualize_detection(img_file, dets, thresh=0.6):
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(img_file)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    num_detections = 0
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        if score < thresh:
            continue
        num_detections += 1
        cls_id = int(klass)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        plt.gca().add_patch(rect)
        class_name = str(cls_id)
        print("{},{}".format(class_name, score))
        plt.gca().text(
            xmin,
            ymin - 2,
            "{:s} {:.3f}".format(class_name, score),
            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            fontsize=12,
            color="white",
        )

    print("Number of detections: " + str(num_detections))
    plt.show()


for i in range(5):
    num = str(i + 1)
    with open(f"./inference_data/face{num}.jpeg", "rb") as image:
        f = image.read()
        b = bytearray(f)
        endpoint_response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="image/jpeg", Body=b)
        results = endpoint_response["Body"].read()
        detections = json.loads(results)
        print(detections["prediction"][0])
        # visualize_detection(f"./inference_data/face{num}.jpeg", detections["prediction"], 0.25)
