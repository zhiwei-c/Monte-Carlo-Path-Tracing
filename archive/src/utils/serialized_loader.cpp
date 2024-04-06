#include "serialized_loader.hpp"

#include <algorithm>
#include <cstdlib>

#include <zlib.h>

#include "../math/coordinate.hpp"
#include "../shapes/triangle.hpp"

NAMESPACE_BEGIN(raytracer)

/// Buffer size used to communicate with zlib. The larger, the better.
static constexpr size_t kZstreamBufferSize = 32768;

static FILE *stream;
static uint8_t m_inflateBuffer[kZstreamBufferSize];
static z_stream m_deflateStream, m_inflateStream;

bool IsNotLittleEndian()
{
    union
    {
        uint8_t charValue[2];
        uint16_t shortValue;
    };

    charValue[0] = 1;
    charValue[1] = 0;

    return shortValue != 1;
}
static const bool not_little_endian = IsNotLittleEndian();

void ParseSerialized(const std::string &filename, int shape_index, std::vector<dvec3> *positions, std::vector<dvec3> *normals,
                     std::vector<dvec2> *texcoords, std::vector<u32vec3> *indices);
void ProcessOffset(int index, uint16_t version);
void InitZlib();

template <typename T>
inline T SwapEndianness(T value);

void ReadFile(void *ptr, size_t size);
void Seek(size_t pos);
size_t GetPos();
size_t GetSize();
uint16_t ReadUint16();
uint32_t ReadUint32();
uint64_t ReadUint64();

void ReadCompressedFile(void *ptr, size_t size);
uint32_t ReadCompressedUint32();
uint64_t ReadCompressedUint64();
std::string ReadCompressedString();
void ReadUint32Array(uint32_t *data, size_t size);
void ReadSingleArray(float *data, size_t size);
void ReadDoubleArray(double *data, size_t size);

std::vector<u32vec3> ReadU32vec3Array(size_t element_num);
std::vector<dvec2> ReadDvec2Array(size_t element_num, bool double_precision);
std::vector<dvec3> ReadDvec3Array(size_t element_num, bool double_precision);

std::vector<Shape *> LoadObject(const std::string &filename, int shape_index, bool face_normals, bool flip_normals,
                                bool flip_tex_coords, const std::string &id, const dmat4 &to_world)
{
    std::vector<dvec3> positions, normals;
    std::vector<dvec2> texcoords;
    std::vector<u32vec3> indices;
    ParseSerialized(filename, shape_index, &positions, &normals, &texcoords, &indices);

    for (dvec3 &position : positions)
    {
        position = TransfromPoint(to_world, position);
    }

    dmat4 normal_to_world = glm::inverse(glm::transpose(to_world));
    for (dvec3 &normal : normals)
    {
        normal = TransfromVec(normal_to_world, normal);
    }

    auto shapes = std::vector<Shape *>(indices.size());
    for (size_t face_index = 0; face_index < indices.size(); ++face_index)
    {
        const uint32_t &v1 = indices[face_index][0], &v2 = indices[face_index][1], &v3 = indices[face_index][2];
        std::vector<dvec3> local_positions = {positions[v1], positions[v2], positions[v3]};

        auto local_normals = std::vector<dvec3>(3);

        auto local_texcoords = std::vector<dvec2>(3, {0, 0});
        if (!texcoords.empty())
        {
            local_texcoords = {texcoords[v1], texcoords[v2], texcoords[v3]};
        }

        if (face_normals || normals.empty())
        {
            const dvec3 v0v1 = local_positions[1] - local_positions[0],
                        v0v2 = local_positions[2] - local_positions[0],
                        normal = glm::normalize(glm::cross(v0v1, v0v2));
            local_normals = std::vector<dvec3>(3, normal);
        }
        else
        {
            local_normals = {normals[v1], normals[v2], normals[v3]};
        }

        auto local_tangents = std::vector<dvec3>(3, {1, 0, 0});
        auto local_bitangents = std::vector<dvec3>(3);
        for (int v = 0; v < 3; ++v)
        {
            if (std::abs(glm::dot(local_normals[v], dvec3{0, 1, 0})) != 1.0)
            {
                local_tangents[v] = glm::normalize(glm::cross(dvec3{0, 1, 0}, local_normals[v]));
            }
            local_bitangents[v] = glm::normalize(glm::cross(local_normals[v], local_tangents[v]));
        }
        shapes[face_index] = new Triangle(id + "_" + std::to_string(face_index), local_positions, local_normals,
                                          local_tangents, local_bitangents, local_texcoords, flip_normals);
    }
    return shapes;
}

void ParseSerialized(const std::string &filename, int shape_index, std::vector<dvec3> *positions, std::vector<dvec3> *normals,
                     std::vector<dvec2> *texcoords, std::vector<u32vec3> *indices)
{
    stream = fopen(filename.c_str(), "rb");
    if (stream == nullptr)
    {
        std::cerr << "[error] open file \"" << filename << "\"failed.\n";
        exit(1);
    }

    uint16_t format = ReadUint16();
    if (format != 0x041C)
    {
        std::cerr << "[error] invalid file format.\n";
        exit(1);
    }

    uint16_t version = ReadUint16();
    if (version != 0x0003 && format != 0x0004)
    {
        std::cerr << "[error] invalid file version.\n";
        exit(1);
    }

    ProcessOffset(shape_index, version);
    InitZlib();

    uint32_t flags = ReadCompressedUint32();

    std::string name;
    if (version == 0x0004)
    {
        name = ReadCompressedString();
    }

    uint64_t vertex_count = ReadCompressedUint64();
    uint64_t triangle_count = ReadCompressedUint64();

    bool double_precision = flags & 0x2000;

    *positions = ReadDvec3Array(vertex_count, double_precision);

    if (flags & 0x0001)
    {
        *normals = ReadDvec3Array(vertex_count, double_precision);
    }

    if (flags & 0x0002)
    {
        *texcoords = ReadDvec2Array(vertex_count, double_precision);
    }

    if (flags & 0x0008)
    {
        std::vector<dvec3> colors = ReadDvec3Array(vertex_count, double_precision);
    }

    *indices = ReadU32vec3Array(triangle_count);

    fclose(stream);
}

void ProcessOffset(int index, uint16_t version)
{
    if (index == 0)
    {
        return;
    }

    size_t offset = 0;

    const size_t stream_size = GetSize();

    /* Determine the position of the requested substream. This is stored
    at the end of the file */
    Seek(stream_size - sizeof(uint32_t));

    uint32_t count = ReadUint32();
    if (index < 0 || index > static_cast<int>(count))
    {
        std::cerr << "[error] Unable to unserialize mesh, shape index is out of range\n";
        fclose(stream);
        exit(1);
    }

    if (version == 0x0004)
    {
        Seek(GetSize() - sizeof(uint64_t) * (count - index) - sizeof(uint32_t));
        offset = ReadUint64();
    }
    else
    {
        Seek(GetSize() - sizeof(uint32_t) * (count - index + 1));
        offset = ReadUint32();
    }

    Seek(offset);
    Seek(GetPos() + sizeof(uint16_t) * 2);
}

void InitZlib()
{
    m_deflateStream.zalloc = Z_NULL;
    m_deflateStream.zfree = Z_NULL;
    m_deflateStream.opaque = Z_NULL;
    int retval = deflateInit2(&m_deflateStream, Z_DEFAULT_COMPRESSION,
                              Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY);
    if (retval != Z_OK)
    {
        std::cerr << "[error] Could not initialize ZLIB: error code \"" << retval << "\"";
        fclose(stream);
        exit(1);
    }

    m_inflateStream.zalloc = Z_NULL;
    m_inflateStream.zfree = Z_NULL;
    m_inflateStream.opaque = Z_NULL;
    m_inflateStream.avail_in = 0;
    m_inflateStream.next_in = Z_NULL;
    retval = inflateInit2(&m_inflateStream, 15);
    if (retval != Z_OK)
    {
        std::cerr << "[error] Could not initialize ZLIB: error code \"" << retval << "\"";
        fclose(stream);
        exit(1);
    }
}

template <typename T>
inline T SwapEndianness(T value)
{
    union
    {
        T value;
        uint8_t byteValue[sizeof(T)];
    } u;

    u.value = value;
    std::reverse(&u.byteValue[0], &u.byteValue[sizeof(T)]);
    return u.value;
}

void ReadFile(void *ptr, size_t size)
{
    size_t bytesRead = fread(ptr, 1, size, stream);
    if (bytesRead != size)
    {
        if (ferror(stream) != 0)
        {
            std::cerr << "[error] Error while reading from file\n";
        }
        else
        {
            std::cerr << "[error] Read less data than expected\n";
        }
        fclose(stream);
        exit(1);
    }
}

void Seek(size_t pos)
{
#ifdef WIN32
    if (_fseeki64(stream, pos, SEEK_SET))
#else
    if (fseeko(stream, (off_t)pos, SEEK_SET))
#endif
    {
        std::cerr << "[error] Error while trying to seek to position \"" << pos
                  << "\" in file";
        fclose(stream);
        exit(1);
    }
}

size_t GetPos()
{
#ifdef WIN32
    long long pos = _ftelli64(stream);
#else
    off_t pos = ftello(stream);
#endif
    if (pos == -1)
    {
        std::cerr << "Error while looking up the position in file\n";
        fclose(stream);
        exit(1);
    }
    return static_cast<size_t>(pos);
}

size_t GetSize()
{
    size_t tmp = GetPos();
    if (fseek(stream, 0, SEEK_END))
    {
        std::cerr << "[error] Error while seeking within file\n";
        fclose(stream);
        exit(1);
    }

    size_t size = GetPos();
#ifdef WIN32
    if (_fseeki64(stream, tmp, SEEK_SET))
#else
    if (fseeko(stream, tmp, SEEK_SET))
#endif
    {
        std::cerr << "[error] Error while seeking within file\n";
        fclose(stream);
        exit(1);
    }
    return size;
}

uint16_t ReadUint16()
{
    uint16_t value;
    ReadFile(&value, sizeof(uint16_t));
    if (not_little_endian)
    {
        value = SwapEndianness(value);
    }
    return value;
}

uint32_t ReadUint32()
{
    uint32_t value;
    ReadFile(&value, sizeof(uint32_t));
    if (not_little_endian)
    {
        value = SwapEndianness(value);
    }
    return value;
}

uint64_t ReadUint64()
{
    uint64_t value;
    ReadFile(&value, sizeof(uint64_t));
    if (not_little_endian)
    {
        value = SwapEndianness(value);
    }
    return value;
}

void ReadCompressedFile(void *ptr, size_t size)
{
    uint8_t *targetPtr = (uint8_t *)ptr;
    while (size > 0)
    {
        if (m_inflateStream.avail_in == 0)
        {
            size_t remaining = GetSize() - GetPos();
            m_inflateStream.next_in = m_inflateBuffer;
            m_inflateStream.avail_in = (uInt)std::min(remaining, sizeof(m_inflateBuffer));
            if (m_inflateStream.avail_in == 0)
            {
                std::cerr << "[error] Read less data than expected\n";
                fclose(stream);
                exit(1);
            }

            ReadFile(m_inflateBuffer, m_inflateStream.avail_in);
        }

        m_inflateStream.avail_out = (uInt)size;
        m_inflateStream.next_out = targetPtr;

        int retval = inflate(&m_inflateStream, Z_NO_FLUSH);
        switch (retval)
        {
        case Z_STREAM_ERROR:
            std::cerr << "[error] inflate(): stream error!\n";
            fclose(stream);
            exit(1);
        case Z_NEED_DICT:
            std::cerr << "[error] inflate(): need dictionary!\n";
            fclose(stream);
            exit(1);
        case Z_DATA_ERROR:
            std::cerr << "[error] inflate(): data error!\n";
            fclose(stream);
            exit(1);
        case Z_MEM_ERROR:
            std::cerr << "[error] inflate(): memory error!\n";
            fclose(stream);
            exit(1);
        };

        size_t outputSize = size - (size_t)m_inflateStream.avail_out;
        targetPtr += outputSize;
        size -= outputSize;

        if (size > 0 && retval == Z_STREAM_END)
        {
            std::cerr << "[error] inflate(): attempting to read past the end of the stream!\n";
            fclose(stream);
            exit(1);
        }
    }
}

uint32_t ReadCompressedUint32()
{
    uint32_t value;
    ReadCompressedFile(&value, sizeof(uint32_t));
    if (not_little_endian)
    {
        value = SwapEndianness(value);
    }
    return value;
}

uint64_t ReadCompressedUint64()
{
    uint64_t value;
    ReadCompressedFile(&value, sizeof(uint64_t));
    if (not_little_endian)
    {
        value = SwapEndianness(value);
    }
    return value;
}

std::string ReadCompressedString()
{
    std::string retval;
    char data;
    do
    {
        ReadCompressedFile(&data, sizeof(char));
        if (data != 0)
        {
            retval += data;
        }
    } while (data != 0);
    return retval;
}

void ReadUint32Array(uint32_t *data, size_t size)
{
    ReadCompressedFile(data, sizeof(uint32_t) * size);
    if (not_little_endian)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = SwapEndianness(data[i]);
        }
    }
}

void ReadSingleArray(float *data, size_t size)
{
    ReadCompressedFile(data, sizeof(float) * size);
    if (not_little_endian)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = SwapEndianness(data[i]);
        }
    }
}

void ReadDoubleArray(double *data, size_t size)
{
    ReadCompressedFile(data, sizeof(double) * size);
    if (not_little_endian)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = SwapEndianness(data[i]);
        }
    }
}

std::vector<u32vec3> ReadU32vec3Array(size_t element_num)
{
    auto data = std::vector<u32vec3>(element_num);
    auto size = element_num * 3;
    uint32_t *temp = new uint32_t[size];
    ReadUint32Array(temp, size);
    for (size_t i = 0; i < element_num; ++i)
    {
        data[i] = {temp[i * 3], temp[i * 3 + 1], temp[i * 3 + 2]};
    }
    delete[] temp;
    return data;
}

std::vector<dvec2> ReadDvec2Array(size_t element_num, bool double_precision)
{
    auto data = std::vector<dvec2>(element_num);
    auto size = element_num * 2;
    if (double_precision)
    {
        double *temp = new double[size];
        ReadDoubleArray(temp, size);
        for (size_t i = 0; i < element_num; ++i)
        {
            data[i] = {temp[i * 2], temp[i * 2 + 1]};
        }
        delete[] temp;
    }
    else
    {
        float *temp = new float[size];
        ReadSingleArray(temp, size);
        for (size_t i = 0; i < element_num; ++i)
        {
            data[i] = {temp[i * 2], temp[i * 2 + 1]};
        }
        delete[] temp;
    }
    return data;
}

std::vector<dvec3> ReadDvec3Array(size_t element_num, bool double_precision)
{
    auto data = std::vector<dvec3>(element_num);
    auto size = element_num * 3;
    if (double_precision)
    {
        double *temp = new double[size];
        ReadDoubleArray(temp, size);
        for (size_t i = 0; i < element_num; ++i)
        {
            data[i] = {temp[i * 3], temp[i * 3 + 1], temp[i * 3 + 2]};
        }
        delete[] temp;
    }
    else
    {
        float *temp = new float[size];
        ReadSingleArray(temp, size);
        for (size_t i = 0; i < element_num; ++i)
        {
            data[i] = {temp[i * 3], temp[i * 3 + 1], temp[i * 3 + 2]};
        }
        delete[] temp;
    }
    return data;
}

NAMESPACE_END(raytracer)