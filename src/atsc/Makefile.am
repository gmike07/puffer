AM_CPPFLAGS = $(CXX17_FLAGS) -I$(srcdir)/../util -I$(srcdir)/../net -I$(srcdir)/../abr -I$(srcdir)/../media-server -I$(srcdir)/../notifier $(libmpeg2_CFLAGS)
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(EXTRA_CXXFLAGS)

bin_PROGRAMS = decoder

decoder_SOURCES = decoder.cc
decoder_LDADD = ../util/libutil.a ../net/libnet.a $(SSL_LIBS) $(libmpeg2_LIBS) -lstdc++fs
