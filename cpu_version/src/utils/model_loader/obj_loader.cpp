#include "obj_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

NAMESPACE_BEGIN(raytracer)

static bool LoadObjEdited(tinyobj::attrib_t *attrib,
                          std::vector<tinyobj::shape_t> *shapes,
                          std::vector<tinyobj::material_t> *materials,
                          std::string *warn,
                          std::string *err,
                          const char *filename,
                          const char *mtl_basedir,
                          bool trianglulate,
                          bool default_vcols_fallback);

static bool LoadObjEdited(tinyobj::attrib_t *attrib,
                          std::vector<tinyobj::shape_t> *shapes,
                          std::vector<tinyobj::material_t> *materials,
                          std::string *warn,
                          std::string *err,
                          std::istream *inStream,
                          tinyobj::MaterialReader *readMatFn /*= NULL*/,
                          bool triangulate,
                          bool default_vcols_fallback);

bool ObjLoader::ParseFromFile(const std::string &filename, const tinyobj::ObjReaderConfig &config)
{
    std::string mtl_search_path;

    if (config.mtl_search_path.empty())
    {
        if (filename.find_last_of("/\\") != std::string::npos)
        {
            mtl_search_path = filename.substr(0, filename.find_last_of("/\\"));
        }
    }
    else
    {
        mtl_search_path = config.mtl_search_path;
    }

    valid_ = LoadObjEdited(&attrib_, &shapes_, &materials_, &warning_, &error_,
                           filename.c_str(), mtl_search_path.c_str(),
                           config.triangulate, config.vertex_color);

    return valid_;
}

bool LoadObjEdited(tinyobj::attrib_t *attrib,
                   std::vector<tinyobj::shape_t> *shapes,
                   std::vector<tinyobj::material_t> *materials,
                   std::string *warn,
                   std::string *err,
                   const char *filename,
                   const char *mtl_basedir,
                   bool trianglulate,
                   bool default_vcols_fallback)
{
    attrib->vertices.clear();
    attrib->normals.clear();
    attrib->texcoords.clear();
    attrib->colors.clear();
    shapes->clear();

    std::stringstream errss;

    std::ifstream ifs(filename);
    if (!ifs)
    {
        errss << "Cannot open file [" << filename << "]" << std::endl;
        if (err)
        {
            (*err) = errss.str();
        }
        return false;
    }

    std::string baseDir = mtl_basedir ? mtl_basedir : "";
    if (!baseDir.empty())
    {
#ifndef _WIN32
        const char dirsep = '/';
#else
        const char dirsep = '\\';
#endif
        if (baseDir[baseDir.length() - 1] != dirsep)
            baseDir += dirsep;
    }
    tinyobj::MaterialFileReader matFileReader(baseDir);

    return LoadObjEdited(attrib, shapes, materials, warn, err, &ifs, &matFileReader,
                         trianglulate, default_vcols_fallback);
}

bool LoadObjEdited(tinyobj::attrib_t *attrib,
                   std::vector<tinyobj::shape_t> *shapes,
                   std::vector<tinyobj::material_t> *materials,
                   std::string *warn,
                   std::string *err,
                   std::istream *inStream,
                   tinyobj::MaterialReader *readMatFn /*= NULL*/,
                   bool triangulate,
                   bool default_vcols_fallback)
{
    std::stringstream errss;

    std::vector<tinyobj::real_t> v;
    std::vector<tinyobj::real_t> vn;
    std::vector<tinyobj::real_t> vt;
    std::vector<tinyobj::real_t> vc;
    std::vector<tinyobj::tag_t> tags;
    tinyobj::PrimGroup prim_group;
    std::string name;

    // material
    std::map<std::string, int> material_map;
    int material = -1;

    // smoothing group id
    unsigned int current_smoothing_id =
        0; // Initial value. 0 means no smoothing.

    int greatest_v_idx = -1;
    int greatest_vn_idx = -1;
    int greatest_vt_idx = -1;

    tinyobj::shape_t shape;

    bool found_all_colors = true;

    size_t line_num = 0;
    std::string linebuf;
    while (inStream->peek() != -1)
    {
        tinyobj::safeGetline(*inStream, linebuf);

        line_num++;

        // Trim newline '\r\n' or '\n'
        if (linebuf.size() > 0)
        {
            if (linebuf[linebuf.size() - 1] == '\n')
                linebuf.erase(linebuf.size() - 1);
        }
        if (linebuf.size() > 0)
        {
            if (linebuf[linebuf.size() - 1] == '\r')
                linebuf.erase(linebuf.size() - 1);
        }

        // Skip if empty line.
        if (linebuf.empty())
        {
            continue;
        }

        // Skip leading space.
        const char *token = linebuf.c_str();
        token += strspn(token, " \t");

        assert(token);
        if (token[0] == '\0')
            continue; // empty line

        if (token[0] == '#')
            continue; // comment line

        // vertex
        if (token[0] == 'v' && IS_SPACE((token[1])))
        {
            token += 2;
            tinyobj::real_t x, y, z;
            tinyobj::real_t r, g, b;

            found_all_colors &= tinyobj::parseVertexWithColor(&x, &y, &z, &r, &g, &b, &token);

            v.push_back(x);
            v.push_back(y);
            v.push_back(z);

            if (found_all_colors || default_vcols_fallback)
            {
                vc.push_back(r);
                vc.push_back(g);
                vc.push_back(b);
            }

            continue;
        }

        // normal
        if (token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2])))
        {
            token += 3;
            tinyobj::real_t x, y, z;
            tinyobj::parseReal3(&x, &y, &z, &token);
            vn.push_back(x);
            vn.push_back(y);
            vn.push_back(z);
            continue;
        }

        // texcoord
        if (token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2])))
        {
            token += 3;
            tinyobj::real_t x, y;
            tinyobj::parseReal2(&x, &y, &token);
            vt.push_back(x);
            vt.push_back(y);
            continue;
        }

        // line
        if (token[0] == 'l' && IS_SPACE((token[1])))
        {
            token += 2;

            tinyobj::__line_t line;

            while (!IS_NEW_LINE(token[0]))
            {
                tinyobj::vertex_index_t vi;
                if (!tinyobj::parseTriple(&token, static_cast<int>(v.size() / 3),
                                          static_cast<int>(vn.size() / 3),
                                          static_cast<int>(vt.size() / 2), &vi))
                {
                    if (err)
                    {
                        std::stringstream ss;
                        ss << "Failed parse `l' line(e.g. zero value for vertex index. "
                              "line "
                           << line_num << ".)\n";
                        (*err) += ss.str();
                    }
                    return false;
                }

                line.vertex_indices.push_back(vi);

                size_t n = strspn(token, " \t\r");
                token += n;
            }

            prim_group.lineGroup.push_back(line);

            continue;
        }

        // points
        if (token[0] == 'p' && IS_SPACE((token[1])))
        {
            token += 2;

            tinyobj::__points_t pts;

            while (!IS_NEW_LINE(token[0]))
            {
                tinyobj::vertex_index_t vi;
                if (!tinyobj::parseTriple(&token, static_cast<int>(v.size() / 3),
                                          static_cast<int>(vn.size() / 3),
                                          static_cast<int>(vt.size() / 2), &vi))
                {
                    if (err)
                    {
                        std::stringstream ss;
                        ss << "Failed parse `p' line(e.g. zero value for vertex index. "
                              "line "
                           << line_num << ".)\n";
                        (*err) += ss.str();
                    }
                    return false;
                }

                pts.vertex_indices.push_back(vi);

                size_t n = strspn(token, " \t\r");
                token += n;
            }

            prim_group.pointsGroup.push_back(pts);

            continue;
        }

        // face
        if (token[0] == 'f' && IS_SPACE((token[1])))
        {
            token += 2;
            token += strspn(token, " \t");

            tinyobj::face_t face;

            face.smoothing_group_id = current_smoothing_id;
            face.vertex_indices.reserve(3);

            while (!IS_NEW_LINE(token[0]))
            {
                tinyobj::vertex_index_t vi;
                if (!tinyobj::parseTriple(&token, static_cast<int>(v.size() / 3),
                                          static_cast<int>(vn.size() / 3),
                                          static_cast<int>(vt.size() / 2), &vi))
                {
                    if (err)
                    {
                        std::stringstream ss;
                        ss << "Failed parse `f' line(e.g. zero value for face index. line "
                           << line_num << ".)\n";
                        (*err) += ss.str();
                    }
                    return false;
                }

                greatest_v_idx = greatest_v_idx > vi.v_idx ? greatest_v_idx : vi.v_idx;
                greatest_vn_idx =
                    greatest_vn_idx > vi.vn_idx ? greatest_vn_idx : vi.vn_idx;
                greatest_vt_idx =
                    greatest_vt_idx > vi.vt_idx ? greatest_vt_idx : vi.vt_idx;

                face.vertex_indices.push_back(vi);
                size_t n = strspn(token, " \t\r");
                token += n;
            }

            // replace with emplace_back + std::move on C++11
            prim_group.faceGroup.push_back(face);

            continue;
        }

        // use mtl
        if ((0 == strncmp(token, "usemtl", 6)) && IS_SPACE((token[6])))
        {
            token += 7;
            std::stringstream ss;
            ss << token;
            std::string namebuf = ss.str();

            int newMaterialId = -1;
            if (material_map.find(namebuf) != material_map.end())
            {
                newMaterialId = material_map[namebuf];
            }
            else
            {
                // { error!! material not found }
                if (warn)
                {
                    (*warn) += "material [ '" + namebuf + "' ] not found in .mtl\n";
                }
            }

            if (newMaterialId != material)
            {
                // Create per-face material. Thus we don't add `shape` to `shapes` at
                // this time.
                // just clear `faceGroup` after `exportGroupsToShape()` call.
                exportGroupsToShape(&shape, prim_group, tags, material, name,
                                    triangulate, v);
                prim_group.faceGroup.clear();
                material = newMaterialId;
            }

            continue;
        }

        // load mtl
        if ((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6])))
        {
            if (readMatFn)
            {
                token += 7;

                std::vector<std::string> filenames;
                tinyobj::SplitString(std::string(token), ' ', filenames);

                if (filenames.empty())
                {
                    if (warn)
                    {
                        std::stringstream ss;
                        ss << "Looks like empty filename for mtllib. Use default "
                              "material (line "
                           << line_num << ".)\n";

                        (*warn) += ss.str();
                    }
                }
                else
                {
                    bool found = false;
                    for (size_t s = 0; s < filenames.size(); s++)
                    {
                        std::string warn_mtl;
                        std::string err_mtl;
                        bool ok = (*readMatFn)(filenames[s].c_str(), materials,
                                               &material_map, &warn_mtl, &err_mtl);
                        if (warn && (!warn_mtl.empty()))
                        {
                            (*warn) += warn_mtl;
                        }

                        if (err && (!err_mtl.empty()))
                        {
                            (*err) += err_mtl;
                        }

                        if (ok)
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        if (warn)
                        {
                            (*warn) +=
                                "Failed to load material file(s). Use default "
                                "material.\n";
                        }
                    }
                }
            }

            continue;
        }

        // group name
        if (token[0] == 'g' && (IS_SPACE((token[1])) || token[1] == '\0'))
        {
            // flush previous face group.
            bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                           triangulate, v);
            (void)ret; // return value not used.

            if (shape.mesh.indices.size() > 0)
            {
                shapes->push_back(shape);
            }

            shape = tinyobj::shape_t();

            // material = -1;
            prim_group.clear();

            std::vector<std::string> names;

            while (!IS_NEW_LINE(token[0]))
            {
                std::string str = tinyobj::parseString(&token);
                names.push_back(str);
                token += strspn(token, " \t\r"); // skip tag
            }

            // names[0] must be 'g'

            if (names.size() < 2)
            {
                // 'g' with empty names
                if (warn)
                {
                    std::stringstream ss;
                    ss << "Empty group name. line: " << line_num << "\n";
                    (*warn) += ss.str();
                    name = "";
                }
            }
            else
            {
                std::stringstream ss;
                ss << names[1];

                // tinyobjloader does not support multiple groups for a primitive.
                // Currently we concatinate multiple group names with a space to get
                // single group name.

                for (size_t i = 2; i < names.size(); i++)
                {
                    ss << " " << names[i];
                }

                name = ss.str();
            }

            continue;
        }

        // object name
        if (token[0] == 'o' && IS_SPACE((token[1])))
        {
            // flush previous face group.
            bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                           triangulate, v);
            (void)ret; // return value not used.

            if (shape.mesh.indices.size() > 0 || shape.lines.indices.size() > 0 ||
                shape.points.indices.size() > 0)
            {
                shapes->push_back(shape);
            }

            // material = -1;
            prim_group.clear();
            shape = tinyobj::shape_t();

            // @todo { multiple object name? }
            token += 2;
            std::stringstream ss;
            ss << token;
            name = ss.str();

            continue;
        }

        if (token[0] == 't' && IS_SPACE(token[1]))
        {
            const int max_tag_nums = 8192; // FIXME(syoyo): Parameterize.
            tinyobj::tag_t tag;

            token += 2;

            tag.name = tinyobj::parseString(&token);

            tinyobj::tag_sizes ts = tinyobj::parseTagTriple(&token);

            if (ts.num_ints < 0)
            {
                ts.num_ints = 0;
            }
            if (ts.num_ints > max_tag_nums)
            {
                ts.num_ints = max_tag_nums;
            }

            if (ts.num_reals < 0)
            {
                ts.num_reals = 0;
            }
            if (ts.num_reals > max_tag_nums)
            {
                ts.num_reals = max_tag_nums;
            }

            if (ts.num_strings < 0)
            {
                ts.num_strings = 0;
            }
            if (ts.num_strings > max_tag_nums)
            {
                ts.num_strings = max_tag_nums;
            }

            tag.intValues.resize(static_cast<size_t>(ts.num_ints));

            for (size_t i = 0; i < static_cast<size_t>(ts.num_ints); ++i)
            {
                tag.intValues[i] = tinyobj::parseInt(&token);
            }

            tag.floatValues.resize(static_cast<size_t>(ts.num_reals));
            for (size_t i = 0; i < static_cast<size_t>(ts.num_reals); ++i)
            {
                tag.floatValues[i] = tinyobj::parseReal(&token);
            }

            tag.stringValues.resize(static_cast<size_t>(ts.num_strings));
            for (size_t i = 0; i < static_cast<size_t>(ts.num_strings); ++i)
            {
                tag.stringValues[i] = tinyobj::parseString(&token);
            }

            tags.push_back(tag);

            continue;
        }

        if (token[0] == 's' && IS_SPACE(token[1]))
        {
            // smoothing group id
            token += 2;

            // skip space.
            token += strspn(token, " \t"); // skip space

            if (token[0] == '\0')
            {
                continue;
            }

            if (token[0] == '\r' || token[1] == '\n')
            {
                continue;
            }

            if (strlen(token) >= 3 && token[0] == 'o' && token[1] == 'f' &&
                token[2] == 'f')
            {
                current_smoothing_id = 0;
            }
            else
            {
                // assume number
                int smGroupId = tinyobj::parseInt(&token);
                if (smGroupId < 0)
                {
                    // parse error. force set to 0.
                    // FIXME(syoyo): Report warning.
                    current_smoothing_id = 0;
                }
                else
                {
                    current_smoothing_id = static_cast<unsigned int>(smGroupId);
                }
            }

            continue;
        } // smoothing group id

        // Ignore unknown command.
    }

    // not all vertices have colors, no default colors desired? -> clear colors
    if (!found_all_colors && !default_vcols_fallback)
    {
        vc.clear();
    }

    if (greatest_v_idx >= static_cast<int>(v.size() / 3))
    {
        if (warn)
        {
            std::stringstream ss;
            ss << "Vertex indices out of bounds (line " << line_num << ".)\n"
               << std::endl;
            (*warn) += ss.str();
        }
    }
    if (greatest_vn_idx >= static_cast<int>(vn.size() / 3))
    {
        if (warn)
        {
            std::stringstream ss;
            ss << "Vertex normal indices out of bounds (line " << line_num << ".)\n"
               << std::endl;
            (*warn) += ss.str();
        }
    }
    if (greatest_vt_idx >= static_cast<int>(vt.size() / 2))
    {
        if (warn)
        {
            std::stringstream ss;
            ss << "Vertex texcoord indices out of bounds (line " << line_num << ".)\n"
               << std::endl;
            (*warn) += ss.str();
        }
    }

    bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                   triangulate, v);
    // exportGroupsToShape return false when `usemtl` is called in the last
    // line.
    // we also add `shape` to `shapes` when `shape.mesh` has already some
    // faces(indices)
    if (ret || shape.mesh.indices.size())
    { // FIXME(syoyo): Support other prims(e.g. lines)
        shapes->push_back(shape);
    }
    prim_group.clear(); // for safety

    if (err)
    {
        (*err) += errss.str();
    }

    attrib->vertices.swap(v);
    attrib->vertex_weights.swap(v);
    attrib->normals.swap(vn);
    attrib->texcoords.swap(vt);
    attrib->texcoord_ws.swap(vt);
    attrib->colors.swap(vc);

    return true;
}

NAMESPACE_END(raytracer)