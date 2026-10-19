/*
 * Copyright (c) 2017 rxi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#if defined(_WIN32)
#include <windows.h>
#endif

#include "microtar.h"

typedef struct {
  char name[100];
  char mode[8];
  char owner[8];
  char group[8];
  char size[12];
  char mtime[12];
  char checksum[8];
  char type;
  char linkname[100];
  char _padding[255];
} mtar_raw_header_t;


static int checked_round_up(unsigned n, unsigned incr, unsigned *result) {
  unsigned remainder;
  unsigned increment;
  if (incr == 0 || result == NULL) {
    return MTAR_EFAILURE;
  }
  remainder = n % incr;
  increment = remainder == 0 ? 0 : incr - remainder;
  if (n > UINT_MAX - increment) {
    return MTAR_EFAILURE;
  }
  *result = n + increment;
  return MTAR_ESUCCESS;
}


static unsigned checksum(const mtar_raw_header_t* rh) {
  unsigned i;
  unsigned char *p = (unsigned char*) rh;
  unsigned res = 256;
  for (i = 0; i < offsetof(mtar_raw_header_t, checksum); i++) {
    res += p[i];
  }
  for (i = offsetof(mtar_raw_header_t, type); i < sizeof(*rh); i++) {
    res += p[i];
  }
  return res;
}


static int parse_octal(const char *field, size_t field_size, unsigned *value) {
  char buffer[13];
  char *cursor;
  char *end;
  unsigned long parsed;
  if (field == NULL || value == NULL || field_size == 0 || field_size >= sizeof(buffer)) {
    return MTAR_EFAILURE;
  }
  memcpy(buffer, field, field_size);
  buffer[field_size] = '\0';
  cursor = buffer;
  while (*cursor == ' ') {
    ++cursor;
  }
  if (*cursor == '\0' || *cursor == '-' || *cursor == '+') {
    return MTAR_EFAILURE;
  }
  errno = 0;
  parsed = strtoul(cursor, &end, 8);
  if (errno == ERANGE || end == cursor || parsed > UINT_MAX) {
    return MTAR_EFAILURE;
  }
  while ((size_t)(end - buffer) < field_size && (*end == '\0' || *end == ' ')) {
    ++end;
  }
  if ((size_t)(end - buffer) < field_size) {
    return MTAR_EFAILURE;
  }
  *value = (unsigned)parsed;
  return MTAR_ESUCCESS;
}


static int tread(mtar_t *tar, void *data, unsigned size) {
  int err;
  if (tar == NULL || tar->read == NULL || (data == NULL && size != 0) || size > UINT_MAX - tar->pos) {
    return MTAR_EREADFAIL;
  }
  err = tar->read(tar, data, size);
  if (err == MTAR_ESUCCESS) {
    tar->pos += size;
  }
  return err;
}


static int twrite(mtar_t *tar, const void *data, unsigned size) {
  int err;
  if (tar == NULL || tar->write == NULL || (data == NULL && size != 0) || size > UINT_MAX - tar->pos) {
    return MTAR_EWRITEFAIL;
  }
  err = tar->write(tar, data, size);
  if (err == MTAR_ESUCCESS) {
    tar->pos += size;
  }
  return err;
}


static int write_null_bytes(mtar_t *tar, int n) {
  int i, err;
  char nul = '\0';
  for (i = 0; i < n; i++) {
    err = twrite(tar, &nul, 1);
    if (err) {
      return err;
    }
  }
  return MTAR_ESUCCESS;
}


static int raw_to_header(mtar_header_t *h, const mtar_raw_header_t *rh) {
  unsigned chksum1, chksum2;

  if (h == NULL || rh == NULL) {
    return MTAR_EFAILURE;
  }

  /* If the checksum starts with a null byte we assume the record is NULL */
  if (*rh->checksum == '\0') {
    return MTAR_ENULLRECORD;
  }

  /* Build and compare checksum */
  chksum1 = checksum(rh);
  if (parse_octal(rh->checksum, sizeof(rh->checksum), &chksum2) != MTAR_ESUCCESS) {
    return MTAR_EBADCHKSUM;
  }
  if (chksum1 != chksum2) {
    return MTAR_EBADCHKSUM;
  }

  /* Load raw header into header */
  memset(h, 0, sizeof(*h));
  if (parse_octal(rh->mode, sizeof(rh->mode), &h->mode) != MTAR_ESUCCESS ||
      parse_octal(rh->owner, sizeof(rh->owner), &h->owner) != MTAR_ESUCCESS ||
      parse_octal(rh->size, sizeof(rh->size), &h->size) != MTAR_ESUCCESS ||
      parse_octal(rh->mtime, sizeof(rh->mtime), &h->mtime) != MTAR_ESUCCESS ||
      memchr(rh->name, '\0', sizeof(rh->name)) == NULL ||
      memchr(rh->linkname, '\0', sizeof(rh->linkname)) == NULL) {
    return MTAR_EFAILURE;
  }
  h->type = rh->type;
  memcpy(h->name, rh->name, sizeof(h->name));
  memcpy(h->linkname, rh->linkname, sizeof(h->linkname));

  return MTAR_ESUCCESS;
}


static int header_to_raw(mtar_raw_header_t *rh, const mtar_header_t *h) {
  unsigned chksum;
  int written;

  if (rh == NULL || h == NULL || memchr(h->name, '\0', sizeof(h->name)) == NULL ||
      memchr(h->linkname, '\0', sizeof(h->linkname)) == NULL) {
    return MTAR_EFAILURE;
  }

  /* Load header into raw header */
  memset(rh, 0, sizeof(*rh));
  written = snprintf(rh->mode, sizeof(rh->mode), "%o", h->mode);
  if (written < 0 || (size_t)written >= sizeof(rh->mode)) return MTAR_EFAILURE;
  written = snprintf(rh->owner, sizeof(rh->owner), "%o", h->owner);
  if (written < 0 || (size_t)written >= sizeof(rh->owner)) return MTAR_EFAILURE;
  written = snprintf(rh->size, sizeof(rh->size), "%o", h->size);
  if (written < 0 || (size_t)written >= sizeof(rh->size)) return MTAR_EFAILURE;
  written = snprintf(rh->mtime, sizeof(rh->mtime), "%o", h->mtime);
  if (written < 0 || (size_t)written >= sizeof(rh->mtime)) return MTAR_EFAILURE;
  rh->type = h->type ? h->type : MTAR_TREG;
  memcpy(rh->name, h->name, strlen(h->name) + 1);
  memcpy(rh->linkname, h->linkname, strlen(h->linkname) + 1);

  /* Calculate and write checksum */
  chksum = checksum(rh);
  written = snprintf(rh->checksum, sizeof(rh->checksum), "%06o", chksum);
  if (written != 6) {
    return MTAR_EFAILURE;
  }
  rh->checksum[7] = ' ';

  return MTAR_ESUCCESS;
}


const char* mtar_strerror(int err) {
  switch (err) {
    case MTAR_ESUCCESS     : return "success";
    case MTAR_EFAILURE     : return "failure";
    case MTAR_EOPENFAIL    : return "could not open";
    case MTAR_EREADFAIL    : return "could not read";
    case MTAR_EWRITEFAIL   : return "could not write";
    case MTAR_ESEEKFAIL    : return "could not seek";
    case MTAR_EBADCHKSUM   : return "bad checksum";
    case MTAR_ENULLRECORD  : return "null record";
    case MTAR_ENOTFOUND    : return "file not found";
  }
  return "unknown error";
}


static int file_write(mtar_t *tar, const void *data, unsigned size) {
  unsigned res = fwrite(data, 1, size, tar->stream);
  return (res == size) ? MTAR_ESUCCESS : MTAR_EWRITEFAIL;
}

static int file_read(mtar_t *tar, void *data, unsigned size) {
  unsigned res = fread(data, 1, size, tar->stream);
  return (res == size) ? MTAR_ESUCCESS : MTAR_EREADFAIL;
}

static int file_seek(mtar_t *tar, unsigned offset) {
  int res = fseek(tar->stream, offset, SEEK_SET);
  return (res == 0) ? MTAR_ESUCCESS : MTAR_ESEEKFAIL;
}

static int file_close(mtar_t *tar) {
  if (tar != NULL && tar->source_type == FROM_FILE && tar->stream != NULL) {
    fclose(tar->stream);
    tar->stream = NULL;
  }
  return MTAR_ESUCCESS;
}


int mtar_open(mtar_t *tar, const char *filename, const char *mode) {
  int err;
  mtar_header_t h;

  if (tar == NULL || filename == NULL || mode == NULL) {
    return MTAR_EOPENFAIL;
  }
  /* Init tar struct and functions */
  memset(tar, 0, sizeof(*tar));
  tar->write = file_write;
  tar->read = file_read;
  tar->seek = file_seek;
  tar->close = file_close;
  tar->source_type = FROM_FILE;
  /* Assure mode is always binary */
  if ( strchr(mode, 'r') ) mode = "rb";
  else if ( strchr(mode, 'w') ) mode = "wb";
  else if ( strchr(mode, 'a') ) mode = "ab";
  else return MTAR_EOPENFAIL;
  /* Open file */
#if defined(_WIN32)
  {
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, NULL, 0);
    wchar_t* wide_filename;
    const wchar_t* wide_mode = *mode == 'r' ? L"rb" : (*mode == 'w' ? L"wb" : L"ab");
    if (length <= 0) return MTAR_EOPENFAIL;
    wide_filename = (wchar_t*)malloc((size_t)length * sizeof(wchar_t));
    if (wide_filename == NULL) return MTAR_EOPENFAIL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, wide_filename, length) == length) {
      tar->stream = _wfopen(wide_filename, wide_mode);
    }
    free(wide_filename);
  }
#else
  tar->stream = fopen(filename, mode);
#endif
  if (!tar->stream) {
    return MTAR_EOPENFAIL;
  }
  /* Read first header to check it is valid if mode is `r` */
  if (*mode == 'r') {
    err = mtar_read_header(tar, &h);
    if (err != MTAR_ESUCCESS) {
      mtar_close(tar);
      return err;
    }
  }

  /* Return ok */
  return MTAR_ESUCCESS;
}


int mtar_close(mtar_t *tar) {
  if (tar == NULL || tar->close == NULL) {
    return MTAR_EFAILURE;
  }
  return tar->close(tar);
}


int mtar_seek(mtar_t *tar, unsigned pos) {
  int err;
  if (tar == NULL || tar->seek == NULL) {
    return MTAR_ESEEKFAIL;
  }
  err = tar->seek(tar, pos);
  if (err == MTAR_ESUCCESS) {
    tar->pos = pos;
  }
  return err;
}


int mtar_rewind(mtar_t *tar) {
  if (tar == NULL) {
    return MTAR_ESEEKFAIL;
  }
  tar->remaining_data = 0;
  tar->last_header = 0;
  return mtar_seek(tar, 0);
}


int mtar_next(mtar_t *tar) {
  int err;
  unsigned padded_size;
  uint64_t next_position;
  mtar_header_t h;
  /* Load header */
  err = mtar_read_header(tar, &h);
  if (err) {
    return err;
  }
  /* Seek to next record */
  err = checked_round_up(h.size, 512, &padded_size);
  if (err != MTAR_ESUCCESS) {
    return err;
  }
  next_position = (uint64_t)tar->pos + sizeof(mtar_raw_header_t) + padded_size;
  if (next_position > UINT_MAX) {
    return MTAR_ESEEKFAIL;
  }
  return mtar_seek(tar, (unsigned)next_position);
}


int mtar_find(mtar_t *tar, const char *name, mtar_header_t *h) {
  int err;
  mtar_header_t header;
  if (tar == NULL || name == NULL) {
    return MTAR_EFAILURE;
  }
  /* Start at beginning */
  err = mtar_rewind(tar);
  if (err) {
    return err;
  }
  /* Iterate all files until we hit an error or find the file */
  while ( (err = mtar_read_header(tar, &header)) == MTAR_ESUCCESS ) {
    if ( !strcmp(header.name, name) ) {
      if (h) {
        *h = header;
      }
      return MTAR_ESUCCESS;
    }
    err = mtar_next(tar);
    if (err != MTAR_ESUCCESS) {
      return err;
    }
  }
  /* Return error */
  if (err == MTAR_ENULLRECORD) {
    err = MTAR_ENOTFOUND;
  }
  return err;
}


int mtar_read_header(mtar_t *tar, mtar_header_t *h) {
  int err;
  mtar_raw_header_t rh;
  if (tar == NULL || h == NULL) {
    return MTAR_EFAILURE;
  }
  /* Save header position */
  tar->last_header = tar->pos;
  /* Read raw header */
  err = tread(tar, &rh, sizeof(rh));
  if (err) {
    return err;
  }
  /* Seek back to start of header */
  err = mtar_seek(tar, tar->last_header);
  if (err) {
    return err;
  }
  /* Load raw header into header struct and return */
  return raw_to_header(h, &rh);
}


int mtar_read_data(mtar_t *tar, void *ptr, unsigned size) {
  int err;
  if (tar == NULL || (ptr == NULL && size != 0)) {
    return MTAR_EREADFAIL;
  }
  /* If we have no remaining data then this is the first read, we get the size,
   * set the remaining data and seek to the beginning of the data */
  if (tar->remaining_data == 0) {
    mtar_header_t h;
    /* Read header */
    err = mtar_read_header(tar, &h);
    if (err) {
      return err;
    }
    /* Seek past header and init remaining data */
    err = mtar_seek(tar, tar->pos + sizeof(mtar_raw_header_t));
    if (err) {
      return err;
    }
    tar->remaining_data = h.size;
  }
  if (size > tar->remaining_data) {
    return MTAR_EREADFAIL;
  }
  /* Read data */
  err = tread(tar, ptr, size);
  if (err) {
    return err;
  }
  tar->remaining_data -= size;
  /* If there is no remaining data we've finished reading and seek back to the
   * header */
  if (tar->remaining_data == 0) {
    return mtar_seek(tar, tar->last_header);
  }
  return MTAR_ESUCCESS;
}


int mtar_write_header(mtar_t *tar, const mtar_header_t *h) {
  mtar_raw_header_t rh;
  int err;
  if (tar == NULL || h == NULL || tar->remaining_data != 0) {
    return MTAR_EWRITEFAIL;
  }
  /* Build raw header and write */
  err = header_to_raw(&rh, h);
  if (err != MTAR_ESUCCESS) {
    return err;
  }
  tar->remaining_data = h->size;
  err = twrite(tar, &rh, sizeof(rh));
  if (err != MTAR_ESUCCESS) {
    tar->remaining_data = 0;
  }
  return err;
}


int mtar_write_file_header(mtar_t *tar, const char *name, unsigned size) {
  mtar_header_t h;
  size_t name_length;
  if (name == NULL) {
    return MTAR_EWRITEFAIL;
  }
  name_length = strlen(name);
  if (name_length >= sizeof(h.name)) {
    return MTAR_EWRITEFAIL;
  }
  /* Build header */
  memset(&h, 0, sizeof(h));
  memcpy(h.name, name, name_length + 1);
  h.size = size;
  h.type = MTAR_TREG;
  h.mode = 0664;
  /* Write header */
  return mtar_write_header(tar, &h);
}


int mtar_write_dir_header(mtar_t *tar, const char *name) {
  mtar_header_t h;
  size_t name_length;
  if (name == NULL) {
    return MTAR_EWRITEFAIL;
  }
  name_length = strlen(name);
  if (name_length >= sizeof(h.name)) {
    return MTAR_EWRITEFAIL;
  }
  /* Build header */
  memset(&h, 0, sizeof(h));
  memcpy(h.name, name, name_length + 1);
  h.type = MTAR_TDIR;
  h.mode = 0775;
  /* Write header */
  return mtar_write_header(tar, &h);
}


int mtar_write_data(mtar_t *tar, const void *data, unsigned size) {
  int err;
  unsigned padded_position;
  if (tar == NULL || (data == NULL && size != 0) || size > tar->remaining_data) {
    return MTAR_EWRITEFAIL;
  }
  /* Write data */
  err = twrite(tar, data, size);
  if (err) {
    return err;
  }
  tar->remaining_data -= size;
  /* Write padding if we've written all the data for this file */
  if (tar->remaining_data == 0) {
    err = checked_round_up(tar->pos, 512, &padded_position);
    if (err != MTAR_ESUCCESS) {
      return err;
    }
    return write_null_bytes(tar, (int)(padded_position - tar->pos));
  }
  return MTAR_ESUCCESS;
}


int mtar_finalize(mtar_t *tar) {
  if (tar == NULL || tar->remaining_data != 0) {
    return MTAR_EWRITEFAIL;
  }
  /* Write two NULL records */
  return write_null_bytes(tar, sizeof(mtar_raw_header_t) * 2);
}


static int memory_read(mtar_t *tar, void *data, unsigned size) {
    if (tar == NULL || (data == NULL && size != 0) || tar->pos > tar->stream_size || size > tar->stream_size - tar->pos) {
        return MTAR_EREADFAIL;
    }
    memcpy(data, (char *)tar->stream + tar->pos, size);
    return MTAR_ESUCCESS;
}

static int memory_seek(mtar_t *tar, unsigned pos) {
    if (tar == NULL || pos > tar->stream_size) {
        return MTAR_ESEEKFAIL;
    }
    return MTAR_ESUCCESS;
}

int mtar_open_memory(mtar_t *tar, void *data, size_t size) {
    if (tar == NULL || data == NULL || size < sizeof(mtar_raw_header_t) || size > UINT_MAX) {
        return MTAR_EOPENFAIL;
    }
    memset(tar, 0, sizeof(*tar));
    tar->read = memory_read;
    tar->seek = memory_seek;
    tar->stream = data;
    tar->stream_size = size;  // Add a field to store the data size
    tar->source_type = FROM_MEMORY;
    tar->close = file_close;

    // Read the first header to verify the data
    mtar_header_t h;
    int err = mtar_read_header(tar, &h);
    if (err != MTAR_ESUCCESS) {
        return err;
    }
    return mtar_rewind(tar);
}
