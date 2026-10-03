import cv2
import numpy as np
import random
import math

WIDTH = 1200
HEIGHT = 800
BGR = (0, 0, 0)
MAX_TRIES = 5000

COLORS = {
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
}

if __name__ == '__main__':


    objects = []

    for shape in ["triangle", "square", "pentagon", "star"]:
        for color_name in ["yellow", "blue", "red"]:
            size = random.randint(40, 75) if shape == "star" else random.randint(35, 70)
            angle = random.uniform(0, 360)

            a = math.radians(angle)
            c = math.cos(a)
            s = math.sin(a)

            if shape == "triangle":
                start = -math.pi / 2
                base_points = np.array(
                    [[size * math.cos(start + 2 * math.pi * i / 3), size * math.sin(start + 2 * math.pi * i / 3)] for i
                     in range(3)], dtype=np.float32)
            elif shape == "square":
                base_points = np.array([[-size, -size], [size, -size], [size, size], [-size, size]], dtype=np.float32)
            elif shape == "pentagon":
                start = -math.pi / 2
                base_points = np.array(
                    [[size * math.cos(start + 2 * math.pi * i / 5), size * math.sin(start + 2 * math.pi * i / 5)] for i
                     in range(5)], dtype=np.float32)
            else:
                start = -math.pi / 2
                pts = []
                for i in range(10):
                    ang = start + math.pi * i / 5
                    rr = size if i % 2 == 0 else size * 0.45
                    pts.append([rr * math.cos(ang), rr * math.sin(ang)])
                base_points = np.array(pts, dtype=np.float32)

            rotated_points = np.array([[x * c - y * s, x * s + y * c] for x, y in base_points], dtype=np.float32)

            x1 = float(np.min(rotated_points[:, 0]))
            y1 = float(np.min(rotated_points[:, 1]))
            x2 = float(np.max(rotated_points[:, 0]))
            y2 = float(np.max(rotated_points[:, 1]))
            shape_w = x2 - x1
            shape_h = y2 - y1

            box_w = int(math.ceil(shape_w + 2 * 12 + 17))
            box_h = int(math.ceil(shape_h + 2 * 12 + 17))

            objects.append({
                "shape": shape,
                "color_name": color_name,
                "color_bgr": COLORS[color_name],
                "local_points": rotated_points,
                "shape_bbox": (x1, y1, x2, y2),
                "box_w": box_w,
                "box_h": box_h,
            })

    random.shuffle(objects)
    placed = []

    for obj in objects:
        ok = False
        for _ in range(MAX_TRIES):
            x = random.randint(0, WIDTH - obj["box_w"])
            y = random.randint(0, HEIGHT - obj["box_h"])
            rect = (x, y, obj["box_w"], obj["box_h"])

            bad = False
            for old in placed:
                ax, ay, aw, ah = rect
                bx, by, bw, bh = old["rect"]
                if not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay):
                    bad = True
                    break

            if bad:
                continue

            x1, y1, x2, y2 = obj["shape_bbox"]
            shape_w = x2 - x1
            shape_h = y2 - y1
            target_x1 = x + (obj["box_w"] - shape_w) / 2
            target_y1 = y + (obj["box_h"] - shape_h) / 2
            dx = target_x1 - x1
            dy = target_y1 - y1

            final_points = obj["local_points"].copy()
            final_points[:, 0] += dx
            final_points[:, 1] += dy

            obj["rect"] = rect
            obj["points"] = final_points
            placed.append(obj)
            ok = True
            break

    image = np.full((HEIGHT, WIDTH, 3), BGR, dtype=np.uint8)



    for obj in placed:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        pts = np.round(obj["points"]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        if 17 % 2 == 0:
            blur_size = 17 + 1
        else:
            blur_size = 17
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        _, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
        layer = np.zeros_like(image)
        layer[:] = obj["color_bgr"]
        image[:] = np.where(mask[:, :, None] == 255, layer, image)

    cv2.imshow("Начальные фигуры", image)
    cv2.waitKey(0)

    red_lower = np.array([0, 0, 50])
    red_higher = np.array([50, 50, 255])
    mask_red = cv2.inRange(image, red_lower, red_higher)

    cv2.imshow("Маска красных ", mask_red)
    cv2.waitKey(0)

    red_objects = cv2.bitwise_and(image, image, mask=mask_red)
    cv2.imshow("красные фигуры", red_objects)
    cv2.waitKey(0)

    gray = cv2.cvtColor(red_objects, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = red_objects.copy()
    epsilon_factor = 0.05

    for cont in contours:

        sm = cv2.arcLength(cont, True)
        apd = cv2.approxPolyDP(cont, epsilon_factor * sm, True)
        if len(apd) == 3:
            cv2.drawContours(result, [cont], -1, (0, 255, 0), 3)

    cv2.imshow("Красные треугольники", result)



    cv2.waitKey(0)
    cv2.destroyAllWindows()