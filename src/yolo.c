#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *voc_names[] = {"frontal face", "profile face", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

void draw_yolo(image im, int num, float thresh, box *boxes, float **probs)
{
    //int classes = 20;
    int classes = 2;
    int i;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            int width = pow(prob, 1./2.)*10+1;
            //width = 8;
            printf("%s: %.2f\n", voc_names[class], prob);
            //class = class * 7 % 20;
			class = class * 7 % 2;
            float red = get_color(0,class,classes);
            float green = get_color(1,class,classes);
            float blue = get_color(2,class,classes);
            //red = green = blue = 0;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
        }
    }
}

void train_yolo(char *cfgfile, char *weightfile, const char* model_dir)
{
    //char *train_images = "data/voc.0712.trainval";
    //char *backup_directory = "/home/pjreddie/backup/";
    char* train_images = "train_images.txt";
	const char* backup_directory = model_dir;
	printf("\nList of Training images: %s\n", train_images);
	printf("Model Directory: %s\n", backup_directory);

    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || i == 600){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_yolo_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    int nms = 0;
    float iou_thresh = .5;
    float nms_thresh = .5;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
        convert_yolo_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms_thresh);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh, char* image_dir, char* image_list, char* output_dir)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	FILE* image_list_file = 0;
	char image_path[1024];
	char image_name[1024];
	char output_name_predict[1024];
	char output_name_resized[1024];
	if (image_list != 0) {
		image_list_file = fopen(image_list, "r");
		if (image_list_file == 0)
			printf("Fail to open image list file: %s\n", image_list);
	}
	int image_count = 0;

	FILE* detect_file = 0;
	char detect_path[1024];
	strcpy(detect_path, "");
	if (output_dir != 0) {
		strcpy(detect_path, output_dir);
		strcat(detect_path, "/");
	}
	strcat(detect_path, "detect.txt");
	detect_file = fopen(detect_path, "w");
	if (detect_file == 0)
		printf("Fail to open detect file: %s\n", detect_path);

	clock_t elapsed_time = 0;
	clock_t end_time = 0;

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
        	if (image_list_file == 0) {
	            printf("Enter Image Path: ");
	            fflush(stdout);
	            input = fgets(input, 256, stdin);
	            if(!input) return;
	            strtok(input, "\n");
        	}
			else {
				strcpy(image_path, "");
				if (image_dir != 0) {
					strcpy(image_path, image_dir);
					strcat(image_path, "/");
				}
				if (!fgets(image_name, 1023 - strlen(image_path), image_list_file))
					break;
				int name_len = strlen(image_name);
				if (image_name[name_len - 2] == '\r')
					image_name[name_len - 2] = '\0';
				else if (image_name[name_len - 1] == '\n')
					image_name[name_len - 1] = '\0';
				strcat(image_path, image_name);
				input = image_path;
			}
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
		//printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		end_time = clock();
		elapsed_time += (end_time - time);
        printf("%s: Predicted in %f seconds.\n", input, sec(end_time - time));

        convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        draw_yolo(im, l.side*l.side*l.n, thresh, boxes, probs);

		strcpy(output_name_predict, "");
		strcpy(output_name_resized, "");
		image_count++;
		if (output_dir != 0) {
			strcpy(output_name_predict, output_dir);
			strcat(output_name_predict, "/");

			strcpy(output_name_resized, output_dir);
			strcat(output_name_resized, "/");
		}
		snprintf(output_name_predict + strlen(output_name_predict), 12, "%d", image_count);
		strcat(output_name_predict, "_detect");
		snprintf(output_name_resized + strlen(output_name_resized), 12, "%d", image_count);
		strcat(output_name_resized, "_resize");

		save_image_jpg(im, output_name_predict);
		//save_image_jpg(sized, output_name_resized);

		if (detect_file != 0) {
			fprintf(detect_file, "%s\n", input);
			int ii;
			int class_num = 2;
			int object_count = 0;
			for (ii = 0; ii < l.side * l.side * l.n; ii++) {
				int class_idx = max_index(probs[ii], class_num);
		        float score = probs[ii][class_idx];
		        if(score > thresh)
					object_count++;
			}
			fprintf(detect_file, "%d\n", object_count);
			for (ii = 0; ii < l.side * l.side * l.n; ii++) {
				int class_idx = max_index(probs[ii], class_num);
		        float score = probs[ii][class_idx];
		        if(score > thresh){
		            box b = boxes[ii];

		            int left  = (b.x-b.w/2.)*im.w;
		            int right = (b.x+b.w/2.)*im.w;
		            int top   = (b.y-b.h/2.)*im.h;
		            int bot   = (b.y+b.h/2.)*im.h;

		            if(left < 0) left = 0;
		            if(right > im.w-1) right = im.w-1;
		            if(top < 0) top = 0;
		            if(bot > im.h-1) bot = im.h-1;

		            fprintf(detect_file, "%d %d %d %d %f\n", left, top,
						right - left + 1, bot - top + 1, score);
		        }
			}
		}

        //show_image(im, "predictions");

        //show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        //cvWaitKey(0);
        //cvDestroyAllWindows();
#endif
        if (filename) break;
    }

	printf("Elapsed time: %d images %f s (%f fps).\n", image_count, sec(elapsed_time),
		image_count / sec(elapsed_time));
	if (image_list_file != 0)
		fclose(image_list_file);
	if (detect_file != 0)
		fclose(detect_file);
}

/*
#ifdef OPENCV
image ipl_to_image(IplImage* src);
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

void demo_swag(char *cfgfile, char *weightfile, float thresh)
{
network net = parse_network_cfg(cfgfile);
if(weightfile){
load_weights(&net, weightfile);
}
detection_layer layer = net.layers[net.n-1];
CvCapture *capture = cvCaptureFromCAM(-1);
set_batch_network(&net, 1);
srand(2222222);
while(1){
IplImage* frame = cvQueryFrame(capture);
image im = ipl_to_image(frame);
cvReleaseImage(&frame);
rgbgr_image(im);

image sized = resize_image(im, net.w, net.h);
float *X = sized.data;
float *predictions = network_predict(net, X);
draw_swag(im, predictions, layer.side, layer.n, "predictions", thresh);
free_image(im);
free_image(sized);
cvWaitKey(10);
}
}
#else
void demo_swag(char *cfgfile, char *weightfile, float thresh){}
#endif
 */

void demo_yolo(char *cfgfile, char *weightfile, float thresh);
#ifndef GPU
void demo_yolo(char *cfgfile, char *weightfile, float thresh){}
#endif

void run_yolo(int argc, char **argv)
{
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
	char* image_dir = find_char_arg(argc, argv, "-imageDir", 0);
	char* image_list = find_char_arg(argc, argv, "-imageList", 0);
	char* output_dir = find_char_arg(argc, argv, "-outputDir", 0);
	fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [model directory (use NONE if not train)] [weights (optional)]\n", argv[0], argv[1]);
    if(argc < 5){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [model directory] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
	char* model_dir = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    //char *filename = (argc > 6) ? argv[6]: 0;
    char *filename = (argc > 6 && image_dir == 0 && image_list == 0) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh, image_dir, image_list, output_dir);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights, model_dir);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo_yolo(cfg, weights, thresh);
}
