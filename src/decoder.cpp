#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"
#include "builtin_interfaces/msg/time.hpp"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

class H264DecoderNode : public rclcpp::Node {
private:
    const AVCodec* codec_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    cv::Mat bgr_frame_; 
    AVBufferRef *hw_device_ctx_ = nullptr;
    enum AVHWDeviceType hw_type_ = AV_HWDEVICE_TYPE_NONE;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    struct QueuedFrame {
        cv::Mat frame;
        builtin_interfaces::msg::Time stamp;  // capture-side stamp, propagated to the output
    };
    std::thread publisher_thread_;
    std::queue<QueuedFrame> frame_publish_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_publisher_thread_{false};
    size_t max_queue_size_ = 10;
    
    int skip_frame_ = 0;
    int frame_counter_ = 0;
    uint64_t decoded_frames_ = 0;
    bool i_frame_only_ = false;
    std::string frame_prefix_;
    std::string decoder_backend_ = "auto";
    std::string publish_lens_ = "full";

    // Crop one lens out of the dual-fisheye frame before publishing. Layout is
    // back|front side by side (front = RIGHT half, same convention as the
    // latency probe's --lens front). Camera-side single-lens encode is not
    // reachable on the X5 (SetActiveSensor drops the USB session), so this
    // downstream crop is how the unused lens stops costing transport: it
    // halves the raw payload (5.53 -> 2.77 MB at 1920x960), worth ~12 ms mean
    // and most of the spike tail on the DDS hop. Returns a view; the caller
    // clones it (which also makes the crop contiguous).
    cv::Mat SelectLens(const cv::Mat& frame) const {
        if (publish_lens_ == "front") {
            return frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
        }
        if (publish_lens_ == "back") {
            return frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
        }
        return frame;
    }

    // Decode-timing instrumentation. Splits the upstream latency budget into
    // "our side of the SDK handoff" (compressed DDS hop + callback queueing +
    // decode + codec buffering, visible as stamp->frame age) vs everything
    // before it (camera encode + USB + SDK internals, which is the remainder).
    uint64_t packets_sent_ = 0;      // packets accepted by avcodec_send_packet
    uint64_t frames_out_ = 0;        // frames returned by avcodec_receive_frame
    double decode_call_ms_sum_ = 0.0;
    double decode_call_ms_max_ = 0.0;
    uint64_t decode_call_count_ = 0;
    double stamp_age_ms_sum_ = 0.0;
    double stamp_age_ms_max_ = 0.0;
    uint64_t stamp_age_count_ = 0;
    std::chrono::steady_clock::time_point last_timing_log_ =
        std::chrono::steady_clock::now();

    void InitFFmpegDecoder() {
        // Backend selection ('decoder' parameter). Output-buffer latency floors
        // measured on the X5 live stream via the "decode timing" log:
        //   libopenh264 - 1 frame (~33 ms; wrapper releases frame N on feeding N+1)
        //   h264 (sw)   - 2 frames (~66 ms), SPS-mandated, cannot go lower
        //   h264_cuvid  - 3 frames (~100 ms) even with LOW_DELAY set
        // "auto" keeps the historical preference: cuvid if usable, else software.
        hw_type_ = AV_HWDEVICE_TYPE_NONE;
        codec_ = nullptr;

        if (decoder_backend_ == "auto" || decoder_backend_ == "cuvid") {
            codec_ = avcodec_find_decoder_by_name("h264_cuvid");
            if (codec_) {
                hw_type_ = AV_HWDEVICE_TYPE_CUDA;
            } else if (decoder_backend_ == "cuvid") {
                RCLCPP_ERROR(this->get_logger(),
                    "decoder:=cuvid requested but h264_cuvid is not in this ffmpeg build; "
                    "falling back to software");
            }
        } else if (decoder_backend_ == "openh264") {
            codec_ = avcodec_find_decoder_by_name("libopenh264");
            if (!codec_) {
                RCLCPP_ERROR(this->get_logger(),
                    "decoder:=openh264 requested but libopenh264 is not in this ffmpeg build; "
                    "falling back to software");
            }
        } else if (decoder_backend_ != "software") {
            RCLCPP_WARN(this->get_logger(),
                "Unknown decoder '%s' (expected auto|cuvid|software|openh264); using software",
                decoder_backend_.c_str());
        }

        if (!codec_) {
            codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
            if (!codec_) {
                RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available");
                return;
            }
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            int err = av_hwdevice_ctx_create(&hw_device_ctx_, hw_type_, nullptr, nullptr, 0);
            if (err < 0) {
                RCLCPP_WARN(this->get_logger(), "Failed to create hardware device context, falling back to software");
                hw_type_ = AV_HWDEVICE_TYPE_NONE;
                codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
                if (!codec_) {
                    RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available");
                    return;
                }
            }
        }

        parser_ctx_ = av_parser_init(codec_->id);
        if (!parser_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE && hw_device_ctx_) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            codec_ctx_->get_format = get_hw_format;
        }

        // Zero cuvid's display-delay buffer (cuviddec.c: ulMaxDisplayDelay =
        // LOW_DELAY ? 0 : 4, i.e. this saves up to 4 frames = ~133 ms at 30
        // fps on the NVDEC path). The software h264 decoder IGNORES this flag
        // (verified against ffmpeg 7.1.1 h264dec.c/h264_slice.c: it re-derives
        // its reorder depth from the SPS at every output decision), so the
        // software path keeps the SPS-mandated buffer - 2 frames / ~66 ms on
        // the X5 live stream - and cannot be forced lower.
        codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open codec");
            CleanupFFmpegDecoder();
            return;
        }
        // The one authoritative line for which backend actually runs - explicit
        // requests can fall back, so don't infer the backend from the request.
        RCLCPP_INFO(this->get_logger(), "Active decoder: %s%s", codec_->name,
            hw_type_ != AV_HWDEVICE_TYPE_NONE ? " (NVDEC)" : "");

        pkt_ = av_packet_alloc();
        if (!pkt_) {
            CleanupFFmpegDecoder();
            return;
        }

        hw_frame_ = av_frame_alloc();
        if (!hw_frame_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            sw_frame_ = av_frame_alloc();
            if (!sw_frame_) {
                CleanupFFmpegDecoder();
                return;
            }
        }
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            QueuedFrame frame_to_publish;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() || stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) {
                    break;
                }
                if (frame_publish_queue_.empty()) {
                    continue;
                }
                frame_to_publish = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame_to_publish.frame.empty() && publisher_) {
                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                std_msgs::msg::Header header;
                // Propagate the capture-side stamp from the compressed frame rather than
                // stamping at publish time. Re-stamping here made the decoded topic's
                // header.stamp ~200 ms late (decode + queue + transport), corrupting any
                // downstream time alignment (VIO, latency matching).
                header.stamp = frame_to_publish.stamp;
                header.frame_id = frame_prefix_ + "camera_frame";
                cv_bridge::CvImage cv_image(header, sensor_msgs::image_encodings::BGR8, frame_to_publish.frame);
                cv_image.toImageMsg(*img_msg);
                publisher_->publish(std::move(img_msg));
            }
        }
    }

    // The capture-side stamp rides through the codec as the packet pts so that
    // each decoded frame is paired with ITS OWN stamp. Pairing the outgoing
    // frame with the stamp of the packet currently being fed is wrong whenever
    // the codec buffers frames (a 2-frame codec buffer mis-stamped every frame
    // 66 ms newer than its true capture).
    static int64_t StampToNs(const builtin_interfaces::msg::Time& t) {
        return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
    }
    static builtin_interfaces::msg::Time NsToStamp(int64_t ns) {
        builtin_interfaces::msg::Time t;
        t.sec = static_cast<int32_t>(ns / 1000000000LL);
        t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
        return t;
    }

    void DecodeAndDisplayPacket(AVPacket* packet, const builtin_interfaces::msg::Time& stamp) {
        const auto call_start = std::chrono::steady_clock::now();
        int ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "avcodec_send_packet failed (ret=%d); packet dropped", ret);
            return;
        }
        ++packets_sent_;

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "avcodec_receive_frame failed (ret=%d)", ret);
                break;
            }
            ++frames_out_;

            // Recover this frame's own capture-side stamp from the pts it
            // carried through the codec; fall back to the current packet's
            // stamp only if the codec dropped it.
            const builtin_interfaces::msg::Time frame_stamp =
                (hw_frame_->pts != AV_NOPTS_VALUE) ? NsToStamp(hw_frame_->pts) : stamp;

            AVFrame* frame_to_display = hw_frame_;

            if (hw_frame_->format == AV_PIX_FMT_CUDA || hw_frame_->format == AV_PIX_FMT_VAAPI) {
                if (av_hwframe_transfer_data(sw_frame_, hw_frame_, 0) < 0) {
                    av_frame_unref(hw_frame_);
                    continue;
                }
                frame_to_display = sw_frame_;
            }

            if (!sws_ctx_ && frame_to_display->width > 0 && frame_to_display->height > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height, (AVPixelFormat)frame_to_display->format,
                    frame_to_display->width, frame_to_display->height, AV_PIX_FMT_BGR24,
                    SWS_POINT, nullptr, nullptr, nullptr);
                
                if (!sws_ctx_) {
                    av_frame_unref(hw_frame_);
                    if (frame_to_display == sw_frame_) av_frame_unref(sw_frame_);
                    break;
                }
                bgr_frame_.create(frame_to_display->height, frame_to_display->width, CV_8UC3);

                // Log the actual delivered (negotiated) resolution. This is the
                // ground truth and may differ from the requested resolution, e.g.
                // the X5 streams a fixed ~2656x1328 over USB regardless of request.
                RCLCPP_INFO(this->get_logger(), "Decoding stream at actual resolution %dx%d.",
                    frame_to_display->width, frame_to_display->height);
            }

            if (sws_ctx_ && !bgr_frame_.empty()) {
                uint8_t* dst_data[4] = { bgr_frame_.data, nullptr, nullptr, nullptr };
                int dst_linesize[4] = { static_cast<int>(bgr_frame_.step[0]), 0, 0, 0 };

                sws_scale(sws_ctx_,
                            (const uint8_t* const*)frame_to_display->data, frame_to_display->linesize,
                            0, frame_to_display->height,
                            dst_data, dst_linesize);

                // Age of the frame at BGR-ready time, relative to its capture-side
                // stamp. Measured before the skip logic so decimation doesn't bias it.
                const double stamp_age_ms =
                    (this->now() - rclcpp::Time(frame_stamp)).seconds() * 1000.0;
                stamp_age_ms_sum_ += stamp_age_ms;
                stamp_age_ms_max_ = std::max(stamp_age_ms_max_, stamp_age_ms);
                ++stamp_age_count_;

                // Apply frame skipping after decoding
                bool should_publish = true;
                
                if (skip_frame_ > 0 && !i_frame_only_) {
                    // Skip frame logic (only when not in i_frame_only mode)
                    should_publish = (frame_counter_++ % (skip_frame_ + 1) == 0);
                }
                
                if (should_publish) {
                    cv::Mat frame_copy = SelectLens(bgr_frame_).clone();
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (frame_publish_queue_.size() < max_queue_size_) {
                            frame_publish_queue_.push({frame_copy, frame_stamp});
                        }
                    }
                    queue_cv_.notify_one();
                    ++decoded_frames_;
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "decoder healthy: %lu frames decoded", (unsigned long)decoded_frames_);
                }
            }
            
            av_frame_unref(hw_frame_);
            if (frame_to_display == sw_frame_) {
                av_frame_unref(sw_frame_);
            }
        }

        const double call_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - call_start).count();
        decode_call_ms_sum_ += call_ms;
        decode_call_ms_max_ = std::max(decode_call_ms_max_, call_ms);
        ++decode_call_count_;
        MaybeLogDecodeTiming();
    }

    // Every ~5 s, report where decode time goes:
    //   call         - synchronous cost of one send_packet+receive_frame pass
    //                  (our decode compute, incl. hw transfer + BGR conversion)
    //   stamp->frame - capture-side stamp to BGR-ready (compressed DDS hop +
    //                  callback queueing + decode + codec buffering); the rest
    //                  of the end-to-end budget is upstream of this node
    //   codec buffer - packets in minus frames out; each buffered frame is a
    //                  hidden frame-period (~33 ms at 30 fps) of latency
    void MaybeLogDecodeTiming() {
        const auto now_sc = std::chrono::steady_clock::now();
        if (now_sc - last_timing_log_ < std::chrono::seconds(5) || decode_call_count_ == 0) {
            return;
        }
        const int64_t codec_buffer =
            static_cast<int64_t>(packets_sent_) - static_cast<int64_t>(frames_out_);
        RCLCPP_INFO(this->get_logger(),
            "decode timing: call avg %.1f ms (max %.1f) over %lu pkts; "
            "stamp->frame avg %.1f ms (max %.1f); codec buffer %ld frames",
            decode_call_ms_sum_ / static_cast<double>(decode_call_count_),
            decode_call_ms_max_, (unsigned long)decode_call_count_,
            stamp_age_count_ ? stamp_age_ms_sum_ / static_cast<double>(stamp_age_count_) : 0.0,
            stamp_age_ms_max_, (long)codec_buffer);
        decode_call_ms_sum_ = 0.0;
        decode_call_ms_max_ = 0.0;
        decode_call_count_ = 0;
        stamp_age_ms_sum_ = 0.0;
        stamp_age_ms_max_ = 0.0;
        stamp_age_count_ = 0;
        last_timing_log_ = now_sc;
    }

    void CleanupFFmpegDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = nullptr;
        }
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
            hw_frame_ = nullptr;
        }
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_close(codec_ctx_); 
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (parser_ctx_) {
            av_parser_close(parser_ctx_);
            parser_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
            hw_device_ctx_ = nullptr;
        }
        codec_ = nullptr;
    }

    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        if (msg->format != "h264") {
            return;
        }

        if (!codec_ctx_ || !pkt_ || !hw_frame_) {
            return;
        }

        // I-frame-only mode still needs the parser to flag keyframes.
        if (i_frame_only_) {
            if (!parser_ctx_) {
                return;
            }
            const uint8_t* cur_data = msg->data.data();
            size_t remaining_size = msg->data.size();
            while (remaining_size > 0) {
                int bytes_parsed = av_parser_parse2(parser_ctx_, codec_ctx_,
                                                    &pkt_->data, &pkt_->size,
                                                    cur_data, static_cast<int>(remaining_size),
                                                    AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
                if (bytes_parsed < 0) {
                    break;
                }
                cur_data += bytes_parsed;
                remaining_size -= bytes_parsed;
                if (pkt_->size > 0 && parser_ctx_->key_frame == 1) {
                    pkt_->pts = StampToNs(msg->header.stamp);
                    DecodeAndDisplayPacket(pkt_, msg->header.stamp);
                }
            }
            return;
        }

        // Normal path: the Insta360 SDK delivers one complete H.264 access unit per
        // OnVideoData callback, which the driver republishes verbatim (insta360_stream.hpp).
        // Feed it straight to the decoder. Running it back through av_parser_parse2 - which
        // buffers a frame waiting for the *next* access unit's start code - starved the
        // decoder: frames went in but ~none came out (0 fps on the decoded topic).
        pkt_->data = const_cast<uint8_t*>(msg->data.data());
        pkt_->size = static_cast<int>(msg->data.size());
        pkt_->pts = StampToNs(msg->header.stamp);
        DecodeAndDisplayPacket(pkt_, msg->header.stamp);
        pkt_->data = nullptr;
        pkt_->size = 0;
        pkt_->pts = AV_NOPTS_VALUE;
    }

public:
    H264DecoderNode() : Node("h264_decoder_node") {
        // Relative defaults so a node namespace (e.g. /cam3) prefixes them and the
        // decoder pairs with the driver running in the same namespace.
        this->declare_parameter("compressed_topic", "dual_fisheye/image/compressed");
        this->declare_parameter("uncompressed_topic", "dual_fisheye/image");
        this->declare_parameter("skip_frame", 0);
        this->declare_parameter("i_frame_only", false);
        this->declare_parameter("frame_prefix", "");
        // Decode backend: auto | cuvid | software | openh264 (see the latency
        // comparison in InitFFmpegDecoder / docs/camera_latency_check.md).
        this->declare_parameter("decoder", "auto");
        // Publish only one lens of the dual-fisheye frame: full | front | back.
        this->declare_parameter("publish_lens", "full");

        std::string subscribe_topic = this->get_parameter("compressed_topic").as_string();
        std::string publish_topic = this->get_parameter("uncompressed_topic").as_string();
        skip_frame_ = this->get_parameter("skip_frame").as_int();
        i_frame_only_ = this->get_parameter("i_frame_only").as_bool();
        frame_prefix_ = this->get_parameter("frame_prefix").as_string();
        decoder_backend_ = this->get_parameter("decoder").as_string();
        publish_lens_ = this->get_parameter("publish_lens").as_string();
        if (publish_lens_ != "full" && publish_lens_ != "front" && publish_lens_ != "back") {
            RCLCPP_WARN(this->get_logger(),
                "Unknown publish_lens '%s' (expected full|front|back); publishing full frame.",
                publish_lens_.c_str());
            publish_lens_ = "full";
        }

        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            subscribe_topic, 10,
            std::bind(&H264DecoderNode::compressed_image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(publish_topic, 10);

        publisher_thread_ = std::thread(&H264DecoderNode::PublisherThreadLoop, this);
        
        InitFFmpegDecoder();

        RCLCPP_INFO(this->get_logger(), "H.264 Decoder Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", subscribe_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: %s", publish_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Skip frame: %d, I-frame only: %s, publish lens: %s",
            skip_frame_, i_frame_only_ ? "true" : "false", publish_lens_.c_str());
    }

    ~H264DecoderNode() {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        CleanupFFmpegDecoder();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<H264DecoderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
