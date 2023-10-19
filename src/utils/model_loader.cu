#include "model_loader.cuh"

#include <algorithm>
#include <thread>
#include <cstdio>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <zlib.h>

#include "math.cuh"

namespace
{
    constexpr size_t kZstreamBufferSize = 32768;

    std::string m_filename;
    FILE *m_stream;
    uint8_t m_inflateBuffer[kZstreamBufferSize];
    z_stream m_deflateStream, m_inflateStream;

    void InitZlib()
    {
        m_deflateStream.zalloc = Z_NULL;
        m_deflateStream.zfree = Z_NULL;
        m_deflateStream.opaque = Z_NULL;
        int retval = deflateInit2(&m_deflateStream, Z_DEFAULT_COMPRESSION,
                                  Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY);
        if (retval != Z_OK)
        {
            fprintf(stderr, "[error] Could not initialize ZLIB: error code '%d'.\n", retval);
            fclose(m_stream);
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
            fprintf(stderr, "[error] Could not initialize ZLIB: error code '%d'.\n", retval);
            fclose(m_stream);
            exit(1);
        }
    }

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
    const bool kNotLittleEndian = IsNotLittleEndian();

    template <typename T>
    T SwapEndianness(T value)
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

    void Seek(size_t pos)
    {
#ifdef WIN32
        if (_fseeki64(m_stream, pos, SEEK_SET))
#else
        if (fseeko(m_stream, (off_t)pos, SEEK_SET))
#endif
        {
            fprintf(stderr, "[error] Error while trying to seek to position '%llu' in file '%s'",
                    pos, m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }
    }

    size_t GetPos()
    {
#ifdef WIN32
        long long pos = _ftelli64(m_stream);
#else
        off_t pos = ftello(m_stream);
#endif
        if (pos == -1)
        {
            fprintf(stderr, "[error] Error while looking up the position in file '%s'.\n",
                    m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }
        return static_cast<size_t>(pos);
    }

    size_t GetSize()
    {
        size_t tmp = GetPos();
        if (fseek(m_stream, 0, SEEK_END))
        {
            fprintf(stderr, "[error] Error while seeking within file '%s'.\n", m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }

        size_t size = GetPos();
#ifdef WIN32
        if (_fseeki64(m_stream, tmp, SEEK_SET))
#else
        if (fseeko(m_stream, tmp, SEEK_SET))
#endif
        {
            fprintf(stderr, "[error] Error while seeking within file '%s'.\n", m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }
        return size;
    }

    void ReadFile(void *ptr, size_t size)
    {
        size_t bytesRead = fread(ptr, 1, size, m_stream);
        if (bytesRead != size)
        {
            if (ferror(m_stream) != 0)
                fprintf(stderr, "[error] Error while reading from file %s.\n", m_filename.c_str());
            else
                fprintf(stderr, "[error] Read less data than expected from file %s.\n",
                        m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }
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
                    fprintf(stderr, "[error] Read less data than expected from file %s.\n",
                            m_filename.c_str());
                    fclose(m_stream);
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
                fprintf(stderr, "[error] inflate(): stream error for file '%s'.\n",
                        m_filename.c_str());
                fclose(m_stream);
                exit(1);
            case Z_NEED_DICT:
                fprintf(stderr, "[error] inflate(): need dictionary for file '%s'.\n",
                        m_filename.c_str());
                fclose(m_stream);
                exit(1);
            case Z_DATA_ERROR:
                fprintf(stderr, "[error] inflate(): data error for file '%s'.\n",
                        m_filename.c_str());
                fclose(m_stream);
                exit(1);
            case Z_MEM_ERROR:
                fprintf(stderr, "[error] inflate(): memory error for file '%s'.\n",
                        m_filename.c_str());
                fclose(m_stream);
                exit(1);
            };

            size_t outputSize = size - (size_t)m_inflateStream.avail_out;
            targetPtr += outputSize;
            size -= outputSize;

            if (size > 0 && retval == Z_STREAM_END)
            {
                fprintf(stderr, "[error] inflate(): attempting to read past the end of the stream");
                fprintf(stderr, " for file '%s'.\n", m_filename.c_str());
                fclose(m_stream);
                exit(1);
            }
        }
    }

    std::string ReadCompressedString()
    {
        std::string retval;
        char data;
        do
        {
            ReadCompressedFile(&data, sizeof(char));
            if (data != 0)
                retval += data;
        } while (data != 0);
        return retval;
    }

    template <typename T>
    T ReadElement(bool compressed)
    {
        T value;
        if (compressed)
            ReadCompressedFile(&value, sizeof(T));
        else
            ReadFile(&value, sizeof(T));
        if (kNotLittleEndian)
            value = SwapEndianness(value);
        return value;
    }

    template <typename T>
    void ReadCompressedArray(T *data, size_t size)
    {
        ReadCompressedFile(data, sizeof(T) * size);
        if (kNotLittleEndian)
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = SwapEndianness(data[i]);
        }
    }

    template <typename VecType, int vec_size, typename ElementType, typename RawType>
    std::vector<VecType> ReadVectorArray(size_t array_size)
    {
        std::vector<VecType> data(array_size);
        const size_t num_emelent = array_size * vec_size;
        std::vector<RawType> raw_data(num_emelent);
        ReadCompressedArray<RawType>(raw_data.data(), num_emelent);
        for (size_t i = 0; i < array_size; ++i)
        {
            for (int j = 0; j < vec_size; ++j)
                data[i][j] = static_cast<ElementType>(raw_data[i * vec_size + j]);
        }
        return data;
    }

    void ProcessOffset(int index_shape, uint16_t version)
    {
        if (index_shape == 0)
            return;

        size_t offset = 0;
        const size_t stream_size = GetSize();

        /* Determine the position of the requested substream. This is stored
        at the end of the file */
        Seek(stream_size - sizeof(uint32_t));

        uint32_t count = ReadElement<uint32_t>(false);
        if (index_shape < 0 || index_shape > static_cast<int>(count))
        {
            fprintf(stderr, "[error] Unable to unserialize mesh, ");
            fprintf(stderr, "shape index is out of range for file '%s'\n", m_filename.c_str());
            fclose(m_stream);
            exit(1);
        }

        if (version == 0x0004)
        {
            Seek(GetSize() - sizeof(uint64_t) * (count - index_shape) - sizeof(uint32_t));
            offset = ReadElement<uint64_t>(false);
        }
        else
        {
            Seek(GetSize() - sizeof(uint32_t) * (count - index_shape + 1));
            offset = ReadElement<uint32_t>(false);
        }

        Seek(offset);
        Seek(GetPos() + sizeof(uint16_t) * 2);
    }

} // namespace

std::vector<Primitive::Info> ModelLoader::Load(const std::string &filename, const Mat4 &to_world,
                                               bool flip_texcoords, bool face_normals,
                                               uint32_t id_bsdf)
{
    fprintf(stderr, "[info] read file \"%s\"\n", filename.c_str());
    int assimp_option = aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_CalcTangentSpace;
    if (!face_normals)
        assimp_option = assimp_option | aiProcess_GenSmoothNormals;
    if (flip_texcoords)
        assimp_option = assimp_option | aiProcess_FlipUVs;
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(filename, assimp_option);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        fprintf(stderr, "[error] ASSIMP:: %s \n", importer.GetErrorString());
        exit(1);
    }
    return ProcessNode(scene, scene->mRootNode, face_normals, to_world, id_bsdf);
}

std::vector<Primitive::Info> ModelLoader::Load(const std::string &filename, int index_shape,
                                               const Mat4 &to_world, bool flip_texcoords,
                                               bool face_normals, uint32_t id_bsdf)
{
    std::vector<Vec3> positions, normals;
    std::vector<Vec2> texcoords;
    std::vector<Uvec3> indices;
    ParseSerialized(filename, index_shape, &positions, &normals, &texcoords, &indices);

    for (Vec3 &position : positions)
        position = TransfromPoint(to_world, position);

    Mat4 normal_to_world = to_world.Transpose().Inverse();
    for (Vec3 &normal : normals)
        normal = TransfromVector(normal_to_world, normal);

    std::vector<Primitive::Info> primitive_info_buffer(indices.size());
    Vertex triangle_vertices[3];
    for (size_t face_index = 0; face_index < indices.size(); ++face_index)
    {
        for (int v = 0; v < 3; ++v)
            triangle_vertices[v].position = positions[indices[face_index][v]];

        if (!texcoords.empty())
        {
            for (int v = 0; v < 3; ++v)
                triangle_vertices[v].texcoord = texcoords[indices[face_index][v]];
        }

        if (face_normals || normals.empty())
        {
            const Vec3 v0v1 = triangle_vertices[1].position - triangle_vertices[0].position,
                       v0v2 = triangle_vertices[2].position - triangle_vertices[0].position,
                       normal = Normalize(Cross(v0v1, v0v2));
            for (int v = 0; v < 3; ++v)
                triangle_vertices[v].normal = normal;
        }
        else
        {
            for (int v = 0; v < 3; ++v)
                triangle_vertices[v].normal = normals[indices[face_index][v]];
        }

        for (int v = 0; v < 3; ++v)
        {
            if (fabs(Dot(triangle_vertices[v].normal, Vec3{0, 1, 0})) < 1.0f)
            {
                triangle_vertices[v].tangent = Normalize(Cross(Vec3{0, 1, 0},
                                                               triangle_vertices[v].normal));
            }
            else
            {
                triangle_vertices[v].tangent = {1, 0, 0};
            }
            triangle_vertices[v].bitangent = Normalize(Cross(triangle_vertices[v].normal,
                                                             triangle_vertices[v].tangent));
        }

        primitive_info_buffer[face_index] = Primitive::Info::CreateTriangle(triangle_vertices,
                                                                            id_bsdf);
    }

    return primitive_info_buffer;
}

std::vector<Primitive::Info> ModelLoader::ProcessNode(const aiScene *scene, aiNode *node,
                                                      bool face_normals, const Mat4 &to_world,
                                                      uint32_t id_bsdf)
{
    std::vector<Primitive::Info> primitive_info_buffer;
    const unsigned int thread_num = std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        unsigned int face_nums = mesh->mNumFaces;

        std::vector<Primitive::Info> local_primitive_info_buffer(face_nums);
        if (face_nums < thread_num)
        {
            ProcessMesh(mesh, 0, face_nums, face_normals, to_world, id_bsdf,
                        local_primitive_info_buffer);
        }
        else
        {
            unsigned int block_length = face_nums / thread_num;
            std::vector<std::thread> workers;
            for (unsigned int j = 0; j < thread_num; ++j)
            {
                unsigned int begin = j * block_length,
                             end = (j == thread_num - 1) ? face_nums : ((j + 1) * block_length);
                workers.push_back(std::thread{ProcessMesh, mesh, begin, end, face_normals, to_world,
                                              id_bsdf, std::ref(local_primitive_info_buffer)});
            }
            for (uint32_t i = 0; i < workers.size(); ++i)
                workers[i].join();
        }
        primitive_info_buffer.insert(primitive_info_buffer.end(),
                                     local_primitive_info_buffer.begin(),
                                     local_primitive_info_buffer.end());
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        std::vector<Primitive::Info> local_primitive_info_buffer = ProcessNode(
            scene, node->mChildren[i], face_normals, to_world, id_bsdf);
        primitive_info_buffer.insert(primitive_info_buffer.end(),
                                     local_primitive_info_buffer.begin(),
                                     local_primitive_info_buffer.end());
    }

    return primitive_info_buffer;
}

void ModelLoader::ProcessMesh(aiMesh *mesh, uint32_t begin, uint32_t end, bool face_normals,
                              const Mat4 &to_world, uint32_t id_bsdf,
                              std::vector<Primitive::Info> &primitive_info_buffer)
{
    const Mat4 normal_to_world = to_world.Transpose().Inverse();
    Vertex triangle_vertices[3];
    for (unsigned int face_index = begin; face_index < end; ++face_index)
    {
        aiFace face = mesh->mFaces[face_index];
        unsigned int indices[3] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};

        for (int v = 0; v < 3; ++v)
        {
            triangle_vertices[v].position = {mesh->mVertices[indices[v]].x,
                                        mesh->mVertices[indices[v]].y,
                                        mesh->mVertices[indices[v]].z};
            triangle_vertices[v].position = TransfromPoint(to_world, triangle_vertices[v].position);
        }

        if (face_normals)
        {
            const Vec3 v0v1 = triangle_vertices[1].position - triangle_vertices[0].position,
                       v0v2 = triangle_vertices[2].position - triangle_vertices[0].position;
            const Vec3 normal = TransfromVector(normal_to_world, Normalize(Cross(v0v1, v0v2)));
            for (int v = 0; v < 3; ++v)
                triangle_vertices[v].normal = normal;
        }
        else
        {
            for (int v = 0; v < 3; ++v)
            {
                triangle_vertices[v].normal = {mesh->mNormals[indices[v]].x,
                                               mesh->mNormals[indices[v]].y,
                                               mesh->mNormals[indices[v]].z};
                triangle_vertices[v].normal = TransfromVector(normal_to_world,
                                                              triangle_vertices[v].normal);
            }
        }

        if (mesh->mTextureCoords[0])
        {
            for (int v = 0; v < 3; ++v)
                triangle_vertices[v].texcoord = {mesh->mTextureCoords[0][indices[v]].x,
                                                 mesh->mTextureCoords[0][indices[v]].y};
        }

        for (int v = 0; v < 3; ++v)
        {
            if (mesh->mTangents)
            {
                triangle_vertices[v].tangent = {mesh->mTangents[indices[v]].x,
                                                mesh->mTangents[indices[v]].y,
                                                mesh->mTangents[indices[v]].z};
                triangle_vertices[v].tangent = TransfromVector(normal_to_world,
                                                               triangle_vertices[v].tangent);
            }
            else if (std::abs(Dot(triangle_vertices[v].normal, Vec3{0, 1, 0})) < 1.0f)
            {
                triangle_vertices[v].tangent = Normalize(Cross(Vec3{0, 1, 0},
                                                               triangle_vertices[v].normal));
            }
            else
            {
                triangle_vertices[v].tangent = {1, 0, 0};
            }

            if (mesh->mBitangents)
            {
                triangle_vertices[v].bitangent = {mesh->mBitangents[indices[v]].x,
                                                  mesh->mBitangents[indices[v]].y,
                                                  mesh->mBitangents[indices[v]].z};
                triangle_vertices[v].bitangent = TransfromVector(normal_to_world,
                                                                 triangle_vertices[v].bitangent);
            }
            else
            {
                triangle_vertices[v].bitangent = Normalize(Cross(triangle_vertices[v].normal,
                                                                 triangle_vertices[v].tangent));
            }
        }

        primitive_info_buffer[face_index] = Primitive::Info::CreateTriangle(triangle_vertices,
                                                                            id_bsdf);
    }
}

void ModelLoader::ParseSerialized(const std::string &filename, int index_shape,
                                  std::vector<Vec3> *positions, std::vector<Vec3> *normals,
                                  std::vector<Vec2> *texcoords, std::vector<Uvec3> *indices)
{
    m_filename = filename;
    m_stream = fopen(filename.c_str(), "rb");
    if (m_stream == nullptr)
    {
        fprintf(stderr, "[error] read file \"%s\" failed.\n", filename.c_str());
        exit(1);
    }

    uint16_t format = ReadElement<uint16_t>(false);
    if (format != 0x041C)
    {
        fprintf(stderr, "[error] invalid file format for \"%s\".\n", filename.c_str());
        exit(1);
    }

    uint16_t version = ReadElement<uint16_t>(false);
    if (version != 0x0003 && format != 0x0004)
    {
        fprintf(stderr, "[error] invalid file version for \"%s\".\n", filename.c_str());
        exit(1);
    }

    ProcessOffset(index_shape, version);
    InitZlib();

    uint32_t flags = ReadElement<uint32_t>(true);

    std::string name;
    if (version == 0x0004)
        name = ReadCompressedString();

    uint64_t vertex_count = ReadElement<uint64_t>(true);
    uint64_t triangle_count = ReadElement<uint64_t>(true);
    bool double_precision = flags & 0x2000;

    *positions = double_precision ? ReadVectorArray<Vec3, 3, float, double>(vertex_count)
                                  : ReadVectorArray<Vec3, 3, float, float>(vertex_count);
    if (flags & 0x0001)
        *normals = double_precision ? ReadVectorArray<Vec3, 3, float, double>(vertex_count)
                                    : ReadVectorArray<Vec3, 3, float, float>(vertex_count);
    if (flags & 0x0002)
        *texcoords = double_precision ? ReadVectorArray<Vec2, 2, float, double>(vertex_count)
                                      : ReadVectorArray<Vec2, 2, float, float>(vertex_count);

    std::vector<Vec3> colors;
    if (flags & 0x0008)
        colors = double_precision ? ReadVectorArray<Vec3, 3, float, double>(vertex_count)
                                  : ReadVectorArray<Vec3, 3, float, float>(vertex_count);

    *indices = ReadVectorArray<Uvec3, 3, unsigned int, uint32_t>(triangle_count);

    fclose(m_stream);
}
