/*
    Copyright (C) Freddy A Cubas "Superstar64"
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/
#include <cmath>
#include <cstddef>
#include <cstring>

#include <algorithm>
#include <memory>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include <cairo.h>
#include <cairo-svg.h>
#include <cairo-ps.h>
#include <cairo-pdf.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

template <typename Array, int length>
constexpr size_t size(Array (&)[length]) {
  return length;
}

struct Point {
  double right;
  double forward;
  double up;
  double w;  // 0 if direction, 1 if position

  using Position = double[3];
  Position& position() { return *reinterpret_cast<Position*>(this); }

  using All = double[4];
  All& all() { return *reinterpret_cast<All*>(this); }
};

double dot(Point left, Point right) {
  double result = 0;

  for (size_t i = 0; i < size(left.position()); i++) {
    result += left.position()[i] * right.position()[i];
  }

  return result;
}

double distance(Point point) {
  double result = 0;
  for (double cord : point.position()) {
    result += cord * cord;
  }
  return sqrt(result);
}

struct Triangle {
  Point points[3];
  Point normal;
  Point pointNormals[3];

  using All = Point[7];
  All& all() { return *reinterpret_cast<All*>(this); }
};

// sort by depth
bool operator<(const Triangle& left, const Triangle& right) {
  return left.points[0].forward + left.points[1].forward + left.points[2].forward < right.points[0].forward + right.points[1].forward + right.points[2].forward;
}

struct Shader {
  virtual void render(cairo_t*, Triangle) = 0;
};

struct EmptyShader : Shader {
  void render(cairo_t*, Triangle) override {}
};

struct SingleColor : Shader {
  double red;
  double green;
  double blue;
  SingleColor(double red, double green, double blue) : red(red), green(green), blue(blue) {}

  void render(cairo_t* context, Triangle triangle) override {
    cairo_set_source_rgb(context, red, green, blue);
    cairo_fill_preserve(context);
  }
};

struct FlatShader : Shader {
  double red;
  double green;
  double blue;
  Point light;
  FlatShader(double red, double green, double blue, Point light) : red(red), green(green), blue(blue), light(light) {}

  void render(cairo_t* context, Triangle triangle) override {
    auto normal = triangle.normal;
    auto cos = dot(light, normal) / (distance(light) * distance(normal));
    auto color = (cos + 1) / 2;

    cairo_set_source_rgb(context, color * red, color * green, color * blue);

    cairo_fill_preserve(context);
  }
};

struct GouraudShader : Shader {
  double red;
  double green;
  double blue;
  Point light;
  GouraudShader(double red, double green, double blue, Point light) : red(red), green(green), blue(blue), light(light) {}

  void render(cairo_t* context, Triangle triangle) override {
    auto pattern = cairo_pattern_create_mesh();
    cairo_mesh_pattern_begin_patch(pattern);
    for (auto& point : triangle.points) {
      cairo_mesh_pattern_line_to(pattern, point.right, point.up);
    }

    int index = 0;
    for (auto& normal : triangle.pointNormals) {
      auto cos = dot(light, normal) / (distance(light) * distance(normal));
      auto color = (cos + 1) / 2;
      cairo_mesh_pattern_set_corner_color_rgb(pattern, index, color * red, color * green, color * blue);
      index++;
    }
    cairo_mesh_pattern_end_patch(pattern);
    cairo_set_source(context, pattern);
    cairo_fill_preserve(context);
    cairo_pattern_destroy(pattern);

    cairo_fill_preserve(context);
  }
};

struct Transform {
  double matrix[4][4];

  static const Transform identity;

  static Transform move(Point point) {
    Transform result = identity;
    for (int i = 0; i < 3; i++) {
      result.matrix[i][3] = point.position()[i];
    }
    return result;
  }

  static Transform rotateByRightAxis(double rad) {
    Transform result = identity;
    result.matrix[1][1] = cos(rad);
    result.matrix[1][2] = -sin(rad);
    result.matrix[2][1] = sin(rad);
    result.matrix[2][2] = cos(rad);
    return result;
  }

  static Transform rotateByUpAxis(double rad) {
    Transform result = identity;
    result.matrix[0][0] = cos(rad);
    result.matrix[0][1] = -sin(rad);
    result.matrix[1][0] = sin(rad);
    result.matrix[1][1] = cos(rad);
    return result;
  }

  static Transform rotateByForwardAxis(double rad) {
    Transform result = identity;
    result.matrix[0][0] = cos(rad);
    result.matrix[0][2] = sin(rad);
    result.matrix[2][0] = -sin(rad);
    result.matrix[2][2] = cos(rad);
    return result;
  }

  static Transform scale(Point vector) {
    Transform result = identity;
    result.matrix[0][0] = vector.position()[0];
    result.matrix[1][1] = vector.position()[1];
    result.matrix[2][2] = vector.position()[2];
    return result;
  }

  Point apply(Point point) {
    Point result;
    static_assert(std::extent<decltype(matrix)>::value == std::extent<typename std::remove_reference<decltype(point.all())>::type>::value,
                  "point and matrix not same size");
    for (size_t y = 0; y < size(point.all()); y++) {
      result.all()[y] = 0;
      for (size_t i = 0; i < size(point.all()); i++) {
        result.all()[y] += matrix[y][i] * point.all()[i];
      }
    }
    return result;
  }
};

const Transform Transform::identity = Transform{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
// matrix muliplication is backwards
// A * B * C is C(B(A))
Transform operator*(Transform left, Transform right) {
  Transform result;
  for (size_t y = 0; y < size(left.matrix); y++) {
    for (size_t x = 0; x < size(left.matrix); x++) {
      result.matrix[y][x] = 0;
      for (size_t i = 0; i < size(left.matrix); i++) {
        result.matrix[y][x] += left.matrix[y][i] * right.matrix[i][x];
      }
    }
  }
  return result;
}

struct Model {
  Triangle* data;
  size_t length;
  Triangle* begin() { return data; }

  Triangle* end() { return data + length; }
};

void calculatePointNormals(Model model) {
  auto hash = [](Point point) {
    size_t result = 0;
    for (size_t i = 0; i < size(point.all()); i++) {
      result ^= std::hash<double>{}(point.all()[i]);
    }
    return result;
  };
  auto equal = [](Point left, Point right) { return std::equal(left.position(), left.position() + size(left.position()), right.position()); };
  std::unordered_map<Point, std::pair<int, Point>, decltype(hash), decltype(equal)> map(1, hash, equal);
  for (size_t i = 0; i < model.length; i++) {
    auto triangle = model.data[i];
    auto normal = triangle.normal;
    for (size_t j = 0; j < size(triangle.pointNormals); j++) {
      map.insert(std::make_pair(triangle.points[j], std::make_pair(0, Point{0, 0, 0, 0})));
      auto& pair = map[triangle.points[j]];
      auto& count = pair.first;
      auto& sum = pair.second;
      count++;
      for (size_t k = 0; k < size(sum.position()); k++) {
        sum.position()[k] += normal.position()[k];
      }
    }
  }
  for (auto& element : map) {
    auto& pair = element.second;
    auto& sum = pair.second;
    auto count = pair.first;
    for (size_t k = 0; k < size(sum.position()); k++) {
      sum.position()[k] /= count;
    }
  }
  for (size_t i = 0; i < model.length; i++) {
    auto& triangle = model.data[i];
    for (size_t j = 0; j < 3; j++) {
      auto& target = triangle.pointNormals[j].position();
      auto& source = map[triangle.points[j]].second.position();
      std::copy(source, source + size(source), target);
    }
  }
}

Point project(Point source, double rightSlope, double upSlope) {
  source.right /= rightSlope * source.forward + 1;
  source.up /= upSlope * source.forward + 1;
  return source;
}

// requires model to be sorted by z axis
Model clip(Model model) {
  auto zero = Point{0, 0, 0};
  auto point = std::lower_bound(model.begin(), model.end(), Triangle{{zero, zero, zero}, zero});
  return Model{point, model.length - (point - model.data)};
}

// modifies model inplace
// requires model to be sorted by z axis
void drawWorld(Model model, cairo_t* context, double rightSlope, double upSlope, bool outline, Shader& shader) {
  for (size_t i = model.length; i != 0; i--) {
    auto triangle = model.data[i - 1];

    for (auto& point : triangle.points) {
      point = project(point, rightSlope, upSlope);
      cairo_line_to(context, point.right, point.up);
    }
    cairo_close_path(context);

    shader.render(context, triangle);

    if (outline) {
      cairo_set_source_rgb(context, 0, 1, 0);
      cairo_stroke_preserve(context);
    }
    cairo_new_path(context);
  }
}

// modifies model inplace
// renders to a cairo context from -1,1 in x and y cordinates
void render(cairo_t* context, Model model, int width, int height, Transform transform, double fow, bool outline, Shader& shader) {
  for (auto& triangle : model) {
    for (auto& point : triangle.all()) {
      point = transform.apply(point);
    }
  }

  std::sort(model.begin(), model.end());
  model = clip(model);

  drawWorld(model, context, tan(fow / 2), tan(fow / 2), outline, shader);
}

void transformView(cairo_t* context, int width, int height) {
  cairo_translate(context, width / 2, height / 2);
  cairo_scale(context, width / 2, -height / 2);
  cairo_set_line_width(context, 1. / 1000);
}

void renderSingle(cairo_surface_t* image, Model model, const char* name, int width, int height, Transform transform, double fow, bool outline, Shader& shader) {
  auto context = cairo_create(image);

  transformView(context, width, height);

  render(context, model, width, height, transform, fow, outline, shader);

  cairo_destroy(context);
}

// ffmpeg boilerplate
struct FFBoilerplate {
  AVFormatContext* avformat_context;
  AVOutputFormat* format;
  AVCodecID codec_id;
  AVCodec* codec;
  AVStream* stream;
  AVCodecContext* avcodec_context;
  int status;
  AVFrame* rgb_frame;
  AVFrame* native_frame;
  SwsContext* converter;
  AVPacket packet;
  int width;
  int height;

  FFBoilerplate(const char* name, int width, int height, int stride, void* data, int frames) {
    this->width = width;
    this->height = height;
    av_register_all();

    avformat_alloc_output_context2(&avformat_context, nullptr, nullptr, name);
    if (!avformat_context) {
      fprintf(stderr, "unknown format\n");
      exit(1);
    }
    avformat_context->duration = frames;

    format = avformat_context->oformat;
    codec_id = format->video_codec;
    codec = avcodec_find_encoder(codec_id);
    if (!codec) {
      fprintf(stderr, "error opening codec\n");
      exit(1);
    }
    stream = avformat_new_stream(avformat_context, nullptr);
    if (!stream) {
      fprintf(stderr, "couldn't open stream\n");
      exit(1);
    }
    stream->id = avformat_context->nb_streams - 1;
    avcodec_context = avcodec_alloc_context3(codec);
    if (!avcodec_context) {
      fprintf(stderr, "couldn't open codec context\n");
      exit(1);
    }
    avcodec_context->bit_rate = 400000;
    avcodec_context->width = width;
    avcodec_context->height = height;
    stream->time_base = AVRational{1, 60};
    avcodec_context->time_base = AVRational{1, 60};
    // avcodec_context->gop = 12;
    avcodec_context->pix_fmt = *codec->pix_fmts;
    status = avcodec_open2(avcodec_context, codec, nullptr);
    if (status < 0) {
      fprintf(stderr, "failed to open codec\n");
      exit(1);
    }
    rgb_frame = av_frame_alloc();
    native_frame = avcodec_context->pix_fmt == AV_PIX_FMT_BGRA ? nullptr : av_frame_alloc();
    rgb_frame->format = AV_PIX_FMT_BGRA;
    rgb_frame->width = width;
    rgb_frame->height = height;
    rgb_frame->data[0] = reinterpret_cast<uint8_t*>(data);
    rgb_frame->linesize[0] = stride;
    rgb_frame->pts = 0;
    if (native_frame) {
      native_frame->format = avcodec_context->pix_fmt;
      native_frame->width = width;
      native_frame->height = height;
      native_frame->pts = 0;
    }
    int status2 = 0;
    if (native_frame) {
      status2 = av_frame_get_buffer(native_frame, 32);
    }
    if (status < 0 || status2 < 0) {
      fprintf(stderr, "couldn't open frame\n");
      exit(1);
    }

    status = avcodec_parameters_from_context(stream->codecpar, avcodec_context);
    if (status < 0) {
      fprintf(stderr, "couldn't copy parameters to stream\n");
      exit(1);
    }
    av_dump_format(avformat_context, 0, name, 1);

    if (!(format->flags & AVFMT_NOFILE)) {
      status = avio_open(&avformat_context->pb, name, AVIO_FLAG_WRITE);
      if (status < 0) {
        fprintf(stderr, "couldn't open file\n");
        exit(0);
      }
    }
    status = avformat_write_header(avformat_context, nullptr);
    if (status < 0) {
      fprintf(stderr, "couldn't write header\n");
      exit(1);
    }
    converter = nullptr;
    if (native_frame) {
      converter = sws_getContext(width, height, AV_PIX_FMT_BGRA, width, height, avcodec_context->pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
    }
  }

  void send_frame_impl(AVFrame* frame) {
    auto status = avcodec_send_frame(avcodec_context, frame);
    if (status < 0) {
      fprintf(stderr, "error encoding video");
      exit(1);
    }
    while (true) {
      av_init_packet(&packet);
      status = avcodec_receive_packet(avcodec_context, &packet);
      if (status < 0) {
        break;
      }
      av_packet_rescale_ts(&packet, avcodec_context->time_base, stream->time_base);
      av_interleaved_write_frame(avformat_context, &packet);
    }
  }

  void send_frame() {
    if (native_frame) {
      sws_scale(converter, rgb_frame->data, rgb_frame->linesize, 0, height, native_frame->data, native_frame->linesize);
    }
    auto frame = native_frame ? native_frame : rgb_frame;
    send_frame_impl(frame);
    frame->pts++;
  }

  ~FFBoilerplate() {
    send_frame_impl(nullptr);
    av_write_trailer(avformat_context);

    sws_freeContext(converter);
    avcodec_free_context(&avcodec_context);
    rgb_frame->data[0] = nullptr;
    av_frame_free(&rgb_frame);
    av_frame_free(&native_frame);

    if (!(format->flags & AVFMT_NOFILE)) {
      avio_closep(&avformat_context->pb);
    }
    avformat_free_context(avformat_context);
  }
};

void renderWebm(Model model, const char* name, int width, int height, Transform transform, double fow, bool outline, Shader& shader, int frames) {
  auto surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
  auto stride = cairo_image_surface_get_stride(surface);
  auto data = cairo_image_surface_get_data(surface);
  FFBoilerplate ffmpeg{name, width, height, stride, data, frames};

  auto context = cairo_create(surface);

  transformView(context, width, height);

  std::vector<Triangle> modelCopyMemory(model.length);
  Model modelCopy{modelCopyMemory.data(), modelCopyMemory.size()};

  for (int i = 0; i < frames; i++) {
    std::copy(model.begin(), model.end(), modelCopy.begin());
    cairo_set_source_rgb(context, 0, 0, 0);
    cairo_paint(context);
    render(context, modelCopy, width, height, transform * Transform::rotateByUpAxis(i * 1.0 / frames * 2 * M_PI), fow, outline, shader);

    cairo_surface_flush(surface);
    ffmpeg.send_frame();
  }

  cairo_destroy(context);
  cairo_surface_destroy(surface);
}

bool startsWith(const char* string, const char* subset) {
  if (strlen(subset) > strlen(string)) {
    return false;
  }
  return std::equal(subset, subset + strlen(subset), string);
}

bool endsWith(const char* string, const char* subset) {
  if (strlen(subset) > strlen(string)) {
    return false;
  }
  return std::equal(string + strlen(string) - strlen(subset), string + strlen(string), subset);
}

void checkEof(FILE* file) {
  if (feof(file)) {
    fprintf(stderr, "early end of input file\n");
    exit(1);
  }
}

// Argument parsing

enum class ArgType { flag, reader };

template <ArgType t>
struct Wrap {
  static constexpr ArgType type = t;
};

template <ArgType type, typename Lambda, typename std::enable_if<type == ArgType::flag>::type* = nullptr>
void callLambdaNoArgs(Lambda lambda) {
  lambda();
}

template <ArgType type, typename Lambda, typename std::enable_if<type != ArgType::flag>::type* = nullptr>
void callLambdaNoArgs(Lambda lambda) {}

template <ArgType type, typename Lambda, typename... Args, typename std::enable_if<type == ArgType::reader>::type* = nullptr>
void callLambdaWithArgs(Lambda lambda, Args... argument) {
  lambda(argument...);
}

template <ArgType type, typename Lambda, typename... Args, typename std::enable_if<type != ArgType::reader>::type* = nullptr>
void callLambdaWithArgs(Lambda lambda, Args... argument) {}

template <typename Remainder>
void matchFull(int& argc, const char**& argv, const char* subset, Remainder) {
  fprintf(stderr, "unknown argument --%s\n", subset);
  exit(1);
}

template <typename Wrap, typename Lambda, typename... T>
void matchFull(int& argc, const char**& argv, const char* subset, char inital, const char* name, Wrap wrap, Lambda lambda, T... rest) {
  if (name != 0 && std::equal(name, name + strlen(name), subset) && (subset[strlen(name)] == 0 || subset[strlen(name)] == '=')) {
    bool equal = subset[strlen(name)] == '=';
    if (wrap.type == ArgType::flag) {
      if (equal) {
        fprintf(stderr, "%s does not expect arguments\n", name);
        exit(1);
      } else {
        callLambdaNoArgs<wrap.type>(lambda);
      }
    } else {
      if (equal) {
        auto value = subset + strlen(name) + 1;
        callLambdaWithArgs<wrap.type>(lambda, value);
      } else {
        argc--;
        argv++;
        if (argc == 0) {
          fprintf(stderr, "%s expects an argument\n", name);
          exit(1);
        }
        callLambdaWithArgs<wrap.type>(lambda, argv[0]);
      }
    }
  } else {
    matchFull(argc, argv, subset, rest...);
  }
}

template <typename Remainder>
void matchSingle(int& argc, const char**& argv, const char*& subset, Remainder) {
  fprintf(stderr, "unknown argument -%s\n", subset);
  exit(1);
}

template <typename Wrap, typename Lambda, typename... T>
void matchSingle(int& argc, const char**& argv, const char*& subset, char inital, const char* name, Wrap wrap, Lambda lambda, T... rest) {
  if (inital != 0 && *subset == inital) {
    subset++;
    if (wrap.type == ArgType::flag) {
      callLambdaNoArgs<wrap.type>(lambda);
    } else {
      if (*subset != 0) {
        callLambdaWithArgs<wrap.type>(lambda, subset);
        subset += strlen(subset);
      } else {
        argc--;
        argv++;
        if (argc == 0) {
          fprintf(stderr, "%s expects an argument\n", name);
          exit(1);
        }
        callLambdaWithArgs<wrap.type>(lambda, argv[0]);
      }
    }
  } else {
    matchSingle(argc, argv, subset, rest...);
  }
}

template <typename Remainder>
void applyRemainder(int& argc, const char**& argv, Remainder remainder) {
  remainder(argv[0]);
}

template <typename Wrap, typename Lambda, typename... T>
void applyRemainder(int& argc, const char**& argv, char inital, const char* name, Wrap wrap, Lambda lambda, T... rest) {
  applyRemainder(argc, argv, rest...);
}

template <typename Wrap, typename Lambda, typename... T>
void parseArgs(int& argc, const char**& argv, char inital, const char* name, Wrap wrap, Lambda lambda, T... rest) {
  if (argc == 0) {
    return;
  }
  if (startsWith(argv[0], "--")) {
    auto subset = argv[0] + 2;
    matchFull(argc, argv, subset, inital, name, wrap, lambda, rest...);
  } else if (startsWith(argv[0], "-") && strlen(argv[0]) > 1) {
    auto subset = argv[0] + 1;
    while (*subset != 0) {
      matchSingle(argc, argv, subset, inital, name, wrap, lambda, rest...);
    }
  } else {
    applyRemainder(argc, argv, inital, name, wrap, lambda, rest...);
  }
  argc--;
  argv++;
  parseArgs(argc, argv, inital, name, wrap, lambda, rest...);
}

double readDouble(const char*& string) {
  char* end;
  auto result = strtod(string, &end);
  if (string == end) {
    fprintf(stderr, "unable to parse double %s\n", string);
    exit(1);
  }
  string = end;
  return result;
}

int readInt(const char*& string) {
  char* end;
  auto result = strtol(string, &end, 10);
  if (string == end) {
    fprintf(stderr, "unable to parse int %s\n", string);
    exit(1);
  }
  string = end;
  return result;
}

enum class ShaderType { empty, single, flat, guorand };

int main(int argc, const char** argv) {
  const char* inputName = nullptr;
  const char* outputName = nullptr;
  bool help = false;
  auto transform = Transform::identity;
  int width = 640;
  int height = 640;
  double fow = 70 * M_PI / 180;
  bool outline = false;
  int frames = 60 * 4;
  double red = 1;
  double green = 1;
  double blue = 1;
  Point light = Point{-1, -1, 1};
  ShaderType shaderType = ShaderType::flat;
  auto selfName = argv[0];
  argc--;
  argv++;
  parseArgs(argc, argv,
            // clang-format off
  '\0', "help", Wrap<ArgType::flag>(), [&help] { help = true; }, 'm', "move", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    transform = Transform::move(Point{readDouble(string), readDouble(string), readDouble(string)}) * transform;
  },
  '\0', "rotate_x", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    transform = Transform::rotateByRightAxis(readDouble(string)) * transform;
  },
  '\0', "rotate_y", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    transform = Transform::rotateByForwardAxis(readDouble(string)) * transform;
  },
  '\0',"rotate_z", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    transform = Transform::rotateByUpAxis(readDouble(string)) * transform;
  },
  's', "scale", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    auto value = readDouble(string);
    transform = Transform::scale(Point{value, value, value}) * transform;
  },
  '\0', "scale_xyz", Wrap<ArgType::reader>(),
  [&transform](const char* string) {
    transform = Transform::scale(Point{readDouble(string), readDouble(string), readDouble(string)}) * transform;
  },
  'w', "width", Wrap<ArgType::reader>(),
  [&width](const char* string) {
    width = readInt(string);
  },
  'h', "height", Wrap<ArgType::reader>(),
  [&height](const char* string) {
    height = readInt(string);
  },
  'f', "fow", Wrap<ArgType::reader>(),
  [&fow](const char* string) {
    fow = readDouble(string);
  },
  'o', "outline", Wrap<ArgType::flag>(),
  [&outline] {
    outline = true;
  },
  'c', "color", Wrap<ArgType::reader>(),
  [&red,&green,&blue](const char* string){
    red = readDouble(string);
    green = readDouble(string);
    blue = readDouble(string);
  },
  'l',"light",Wrap<ArgType::reader>(),
  [&light](const char* string){
    light = Point{readDouble(string),readDouble(string),readDouble(string)};
  },
  'e', "empty", Wrap<ArgType::flag>(),
  [&shaderType] {
    shaderType = ShaderType::empty;
  },
  's', "single", Wrap<ArgType::flag>(),
  [&shaderType] {
    shaderType = ShaderType::single;
  },
  '\0', "flat", Wrap<ArgType::flag>(),
  [&shaderType] {
    shaderType = ShaderType::flat;
  },
  'g', "guorand", Wrap<ArgType::flag>(),
  [&shaderType] {
    shaderType = ShaderType::guorand;
  },
  '\0', "frames", Wrap<ArgType::reader>(),
  [&frames](const char* string) {
    frames = readInt(string);
  },
  [&inputName, &outputName](const char* file) {
    if (inputName == nullptr) {
      inputName = file;
    } else if (outputName == nullptr) {
      outputName = file;
    } else {
      fprintf(stderr, "unknown argument %s\n", file);
      exit(1);
    }
  });
  // clang-format on
  if (help || inputName == nullptr || outputName == nullptr) {
    printf(
        "%s [options] inputFile.stl outputFile.[png,svg,ps,pdf,webm]\n\
    --help            print help message\n\
 -m --move='x y z'    move model to\n\
    --rotate_x='rad'  rotate model by x axis\n\
    --rotate_y='rad'  rotate model by y axis\n\
    --rotate_z='rad'  rotate model by z axis\n\
 -s --scale='mul'     scale model\n\
    --scale_xyz='mul mul mul'\n\
                      scale model parts\n\
 -w --width='width'   set width\n\
 -h --height='height' set height\n\
 -f --fow='rad'       set field of view\n\
 -o   --outline         outline polygons\n\
 -c --color='r g b'   set color(0-1) for shader\
 -l --light='x y z'   set light vector for shader\n\
 -e --empty           empty shading(combine with --outline)\n\
 -s --single          use single color\n\
    --flat            use flat shading(default)\n\
 -g --gourand         use gourand shading\n\
    --frames='frames' how many frame to output in video\n\
 ",
        selfName);
    return 0;
  }
  FILE* file = *inputName == '-' ? stdin : fopen(inputName, "rb");
  if (!file) {
    perror(argv[1]);
    return 1;
  }
  char header[80];
  fread(header, sizeof(header), 1, file);
  checkEof(file);
  uint32_t length = 0;
  fread(&length, sizeof(length), 1, file);
  checkEof(file);
  std::vector<Triangle> triangles(length);
  for (auto& triangle : triangles) {
    checkEof(file);
    float data[4][3];
    fread(data, sizeof(data), 1, file);
    checkEof(file);

    triangle.normal.right = data[0][0];
    triangle.normal.forward = data[0][1];
    triangle.normal.up = data[0][2];
    triangle.normal.w = 0;
    for (int i = 0; i < 3; i++) {
      triangle.points[i].right = data[i + 1][0];
      triangle.points[i].forward = data[i + 1][1];
      triangle.points[i].up = data[i + 1][2];
      triangle.points[i].w = 1;
    }
    uint16_t attributes = 0;
    fread(&attributes, sizeof(attributes), 1, file);
  }
  fclose(file);

  std::unique_ptr<Shader> shaderMemory;
  if (shaderType == ShaderType::empty) {
    shaderMemory = decltype(shaderMemory)(new EmptyShader());
  } else if (shaderType == ShaderType::single) {
    shaderMemory = decltype(shaderMemory)(new SingleColor(red, green, blue));
  } else if (shaderType == ShaderType::flat) {
    shaderMemory = decltype(shaderMemory)(new FlatShader(red, green, blue, light));
  } else {
    shaderMemory = decltype(shaderMemory)(new GouraudShader(red, green, blue, light));
  }
  auto& shader = *shaderMemory.get();

  auto model = Model{triangles.data(), triangles.size()};
  calculatePointNormals(model);
  if (endsWith(outputName, ".png")) {
    auto image = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
    renderSingle(image, model, outputName, width, height, transform, fow, outline, shader);
    cairo_surface_write_to_png(image, outputName);
    cairo_surface_destroy(image);
  } else if (endsWith(outputName, ".svg")) {
    auto image = cairo_svg_surface_create(outputName, width, height);
    renderSingle(image, model, outputName, width, height, transform, fow, outline, shader);
    cairo_surface_destroy(image);
  } else if (endsWith(outputName, ".ps")) {
    auto image = cairo_ps_surface_create(outputName, width, height);
    renderSingle(image, model, outputName, width, height, transform, fow, outline, shader);
    cairo_surface_destroy(image);
  } else if (endsWith(outputName, ".pdf")) {
    auto image = cairo_pdf_surface_create(outputName, width, height);
    renderSingle(image, model, outputName, width, height, transform, fow, outline, shader);
    cairo_surface_destroy(image);
  } else {
    renderWebm(model, outputName, width, height, transform, fow, outline, shader, frames);
  }
}
