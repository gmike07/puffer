AM_CPPFLAGS = -I$(srcdir)/../util $(CXX17_FLAGS) $(SSL_CFLAGS) \
            -isystem$(srcdir)/../../third_party/json.upstream/single_include/nlohmann \
            -isystem$(srcdir)/../../third_party/libtorch/include \
            -isystem$(srcdir)/../../third_party/libtorch/include/torch/csrc/api/include \
            -I$(srcdir)/../abr -I$(srcdir)/../util -I$(srcdir)/../media-server -I$(srcdir)/../notifier \
            -I/usr/include/python3.7 \
            -I/home/ubuntu/.local/lib/python3.7/site-packages/pybind11/include
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(EXTRA_CXXFLAGS) -pthread

noinst_LIBRARIES = libnet.a

# libnet_a_LDFLAGS = -L../../third_party/libtorch/lib \
# 	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib'

# libnet_a_LDADD = $(POSTGRES_LIBS) $(SSL_LIBS) $(CRYPTO_LIBS) $(YAML_LIBS) -lstdc++fs \
#                     -ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread
 

libnet_a_SOURCES = address.cc address.hh body_parser.hh \
                   chunked_parser.cc chunked_parser.hh \
                   http_header.cc http_header.hh \
                   http_message.cc http_message.hh \
                   http_message_sequence.hh http_request.cc http_request.hh \
                   http_request_parser.hh http_response.cc http_response.hh \
                   http_response_parser.cc http_response_parser.hh \
                   mime_type.cc mime_type.hh \
                   secure_socket.cc secure_socket.hh socket.cc socket.hh cc_socket_helper.cc \
                   serialization.cc serialization.hh \
                   strict_conversions.hh strict_conversions.cc \
                   nb_secure_socket.hh nb_secure_socket.cc \
                   ws_frame.hh ws_frame.cc \
                   ws_message.hh ws_message.cc \
                   ws_message_parser.hh ws_message_parser.cc \
                   ws_server.hh ws_server.cc \
                   cc_logging.hh cc_logging.cc \
                   cc_logging_objects.cc cc_logging_objects.hh \
                   cc_logging_functions.hh
