#include "renderer.h"

#define PRINT_PROGRESS 1

__global__ void RenderInit(int max_x, int max_y, int resolution, curandState *rand_state)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y)
        return;

    auto pixel_index = j * max_x + i;
    if (pixel_index >= resolution)
        return;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

#ifdef PRINT_PROGRESS
__global__ void RenderProcess(int width, int height, int resolution, curandState *rand_state, Camera *camera,
                              Integrator *integrator, float *frame_data, volatile int *progress)
#else
__global__ void RenderProcess(int width, int height, int resolution, curandState *rand_state, Camera *camera,
                              Integrator *integrator, float *frame_data)
#endif
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;

#ifdef PRINT_PROGRESS
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        atomicAdd((int *)progress, 1);
        __threadfence_system();
    }
#endif

    if (i >= width || j >= height)
        return;

    // if (i != 450 || j != 105)
    //      return;

    auto pixel_index = j * width + i;

    auto local_rand_state = rand_state[pixel_index];

    auto spp = camera->SampleCount();
    auto spp_x = static_cast<int>(sqrt((Float)spp));
    auto spp_y = static_cast<int>(spp / spp_x);
    auto dxx = 1.0 / spp_x;
    auto dyy = 1.0 / spp_y;
    auto spp_r = spp - spp_x * spp_y;
    auto offset = vec2(0);
    auto look_dir = vec3(0);

    auto col = vec3(0, 0, 0);

    for (int k_x = 0; k_x < spp_x; k_x++)
    {
        for (int k_y = 0; k_y < spp_y; k_y++)
        {
            offset.x = (k_x + curand_uniform(&local_rand_state)) * dxx;
            offset.y = (k_y + curand_uniform(&local_rand_state)) * dyy;
            look_dir = camera->GetDirection(i, j, offset);
            col += integrator->Shade(camera->EyePosition(), look_dir, &local_rand_state);
        }
    }

    for (int s = 0; s < spp_r; s++)
    {
        offset.x = curand_uniform(&local_rand_state);
        offset.y = curand_uniform(&local_rand_state);
        look_dir = camera->GetDirection(i, j, offset);
        col += integrator->Shade(camera->EyePosition(), look_dir, &local_rand_state);
    }

    col /= static_cast<Float>(spp);
    for (int i = 0; i < 3; i++)
    {
        if (col[i] < 0)
            col[i] = 0;
        col[i] = ApplyGamma(col[i], camera->GammaInv());
        frame_data[pixel_index * 3 + i] = col[i];
    }
}

Renderer::~Renderer()
{
    CheckCudaErrors(cudaDeviceSynchronize());
    for (auto &texture_info : texture_info_list_)
    {
        delete texture_info;
        texture_info = nullptr;
    }

    for (auto &shape_info : shape_info_list_)
    {
        delete shape_info;
        shape_info = nullptr;
    }

    if (env_map_info_)
    {
        delete env_map_info_;
        env_map_info_ = nullptr;
    }

    CheckCudaErrors(cudaGetLastError());

    for (auto &texture_bitmap_data : texture_bitmap_data_)
        CheckCudaErrors(cudaFree(texture_bitmap_data));

    for (auto &bvhnode : bvhnode_list_)
        CheckCudaErrors(cudaFree(bvhnode));

    FreeBsdfs<<<1, 1>>>(bsdf_info_list_.size(), bsdf_list_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    CheckCudaErrors(cudaFree(texture_list_));
    CheckCudaErrors(cudaFree(mesh_list_));
    CheckCudaErrors(cudaFree(bsdf_list_));
    CheckCudaErrors(cudaFree(shapebvh_list_));
    CheckCudaErrors(cudaFree(scenebvh_node_list_));
    CheckCudaErrors(cudaFree(scenebvh_));
    CheckCudaErrors(cudaFree(integrator_));
    CheckCudaErrors(cudaFree(camera_));
    CheckCudaErrors(cudaFree(emitter_idx_list_));
    CheckCudaErrors(cudaFree(env_map_));
    cudaDeviceReset();
}

void Renderer::InitTextureList()
{
    auto texture_num = texture_info_list_.size();
    texture_list_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&texture_list_, texture_num * sizeof(Texture)));
    for (uint texture_idx = 0; texture_idx < texture_num; texture_idx++)
    {
        switch (texture_info_list_[texture_idx]->type)
        {
        case kConstant:
        {
            auto color = texture_info_list_[texture_idx]->color;
            InitConstantTexture<<<1, 1>>>(texture_idx, color, texture_list_);
            CheckCudaErrors(cudaGetLastError());
            CheckCudaErrors(cudaDeviceSynchronize());
            break;
        }
        case kBitmap:
        {
            auto resolution = texture_info_list_[texture_idx]->colors.size();
            float *bitmap_data = nullptr;
            CheckCudaErrors(cudaMallocManaged(&bitmap_data, resolution * sizeof(float)));
            cudaMemcpy(bitmap_data, texture_info_list_[texture_idx]->colors.data(), resolution * sizeof(float),
                       cudaMemcpyHostToDevice);
            texture_bitmap_data_.push_back(bitmap_data);
            InitBitmapTexture<<<1, 1>>>(texture_idx, texture_info_list_[texture_idx]->width,
                                        texture_info_list_[texture_idx]->height, texture_info_list_[texture_idx]->channel,
                                        bitmap_data, texture_list_);
            CheckCudaErrors(cudaGetLastError());
            CheckCudaErrors(cudaDeviceSynchronize());
            break;
        }
        default:
            PrintExcuError();
            break;
        }
    }
}

void Renderer::InitBsdfList()
{
    auto bsdf_num = bsdf_info_list_.size();
    bsdf_list_ = nullptr;
    CheckCudaErrors(cudaMalloc(&bsdf_list_, bsdf_num * sizeof(Bsdf *)));
    BsdfInfo *local_bsdf_info_list = nullptr;
    CheckCudaErrors(cudaMallocManaged(&local_bsdf_info_list, bsdf_num * sizeof(BsdfInfo)));
    cudaMemcpy(local_bsdf_info_list, bsdf_info_list_.data(), bsdf_num * sizeof(BsdfInfo), cudaMemcpyHostToDevice);

    CreateBsdfs<<<1, 1>>>(bsdf_num, local_bsdf_info_list, texture_list_, bsdf_list_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(local_bsdf_info_list));
    local_bsdf_info_list = nullptr;
}

void Renderer::InitVertexIndexBuffer(Vertex *&vertex_list, uvec3 *&mesh_idx_list, uint *&mesh_bsdf_idx_list, uint &mesh_num,
                                     std::vector<uvec2> &mesh_idx_range_list)
{
    std::cerr << "[info] load shapes ..." << std::endl;
    auto local_vertex_list = std::vector<Vertex>();
    auto local_mesh_idx_list = std::vector<uvec3>();
    auto local_mesh_bsdf_idx_list = std::vector<uint>();
    mesh_idx_range_list = std::vector<uvec2>();
    for (auto &shape_info : shape_info_list_)
    {
        auto old_i_num = local_mesh_idx_list.size();
        auto bump_mapping = (bsdf_info_list_[shape_info->bsdf_idx].bump_map_idx != kUintMax);
        switch (shape_info->type)
        {
        case kCube:
            LoadCube(shape_info, bump_mapping, local_vertex_list, local_mesh_idx_list);
            bsdf_info_list_[shape_info->bsdf_idx].twosided = true;
            break;
        case kDisk:
            LoadDisk(shape_info, bump_mapping, local_vertex_list, local_mesh_idx_list);
            break;
        case kMeshes:
            LoadMeshes(shape_info, bump_mapping, local_vertex_list, local_mesh_idx_list);
            break;
        case kRectangle:
            LoadRectangle(shape_info, bump_mapping, local_vertex_list, local_mesh_idx_list);
            break;
        case kSphere:
            LoadSphere(shape_info, bump_mapping, local_vertex_list, local_mesh_idx_list);
            bsdf_info_list_[shape_info->bsdf_idx].twosided = true;
            break;
        default:
            PrintExcuError();
            break;
        }
        auto new_i_num = local_mesh_idx_list.size();
        local_mesh_bsdf_idx_list.reserve(new_i_num);
        local_mesh_bsdf_idx_list.insert(local_mesh_bsdf_idx_list.end(), new_i_num - old_i_num, shape_info->bsdf_idx);
        mesh_idx_range_list.emplace_back(old_i_num, new_i_num);
    }
    mesh_num = local_mesh_idx_list.size();
    vertex_list = nullptr;
    CheckCudaErrors(cudaMallocManaged((void **)&vertex_list, local_vertex_list.size() * sizeof(Vertex)));
    cudaMemcpy(vertex_list, local_vertex_list.data(), local_vertex_list.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
    mesh_idx_list = nullptr;
    CheckCudaErrors(cudaMallocManaged((void **)&mesh_idx_list, mesh_num * sizeof(uvec3)));
    cudaMemcpy(mesh_idx_list, local_mesh_idx_list.data(), mesh_num * sizeof(uvec3), cudaMemcpyHostToDevice);
    mesh_bsdf_idx_list = nullptr;
    CheckCudaErrors(cudaMallocManaged((void **)&mesh_bsdf_idx_list, mesh_num * sizeof(uint)));
    cudaMemcpy(mesh_bsdf_idx_list, local_mesh_bsdf_idx_list.data(), mesh_num * sizeof(uint), cudaMemcpyHostToDevice);

    timer_.PrintTimePassed("load shape");
}

void Renderer::InitShapesMeshes(std::vector<uvec2> &mesh_idx_range_list,
                                std::vector<AABB> &mesh_aabb_list,
                                std::vector<Float> &mesh_area_list)
{
    std::cerr << "[info] create mesh ..." << std::endl;
    Vertex *vertex_list = nullptr;
    uvec3 *mesh_idx_list = nullptr;
    uint *mesh_bsdf_idx_list = nullptr;
    uint mesh_num = 0;
    mesh_idx_range_list = std::vector<uvec2>();
    InitVertexIndexBuffer(vertex_list, mesh_idx_list, mesh_bsdf_idx_list, mesh_num, mesh_idx_range_list);

    InitTextureList();
    InitBsdfList();
    //
    mesh_list_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&mesh_list_, mesh_num * sizeof(Shape)));
    AABB *local_mesh_aabbs = nullptr;
    CheckCudaErrors(cudaMallocManaged(&local_mesh_aabbs, mesh_num * sizeof(Shape)));
    Float *local_mesh_areas = nullptr;
    CheckCudaErrors(cudaMallocManaged(&local_mesh_areas, mesh_num * sizeof(Float)));
    auto nx = static_cast<uint>(sqrt(mesh_num) + 1);
    auto ny = static_cast<uint>(mesh_num / nx + 1);
    auto blocks = dim3(nx / 16 + 1, ny / 16 + 1);
    auto threads = dim3(16, 16);
    CreateMeshes<<<blocks, threads>>>(nx, ny, mesh_num, vertex_list, mesh_idx_list, mesh_bsdf_idx_list, bsdf_list_,
                                      local_mesh_aabbs, local_mesh_areas, mesh_list_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    mesh_aabb_list = std::vector<AABB>(mesh_num);
    cudaMemcpy(&mesh_aabb_list[0], local_mesh_aabbs, mesh_num * sizeof(AABB), cudaMemcpyDeviceToHost);

    mesh_area_list = std::vector<Float>(mesh_num);
    cudaMemcpy(&mesh_area_list[0], local_mesh_areas, mesh_num * sizeof(Float), cudaMemcpyDeviceToHost);

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(vertex_list));
    CheckCudaErrors(cudaFree(mesh_idx_list));
    CheckCudaErrors(cudaFree(mesh_bsdf_idx_list));
    CheckCudaErrors(cudaFree(local_mesh_aabbs));
    CheckCudaErrors(cudaFree(local_mesh_areas));
    vertex_list = nullptr;
    mesh_idx_list = nullptr;
    mesh_bsdf_idx_list = nullptr;
    local_mesh_aabbs = nullptr;
    local_mesh_areas = nullptr;

    timer_.PrintTimePassed("create mesh ");
}

void Renderer::InitShapeBvh(std::vector<BvhNodeInfo> &shape_info_list)
{
    auto mesh_idx_range_list = std::vector<uvec2>();
    auto scene_mesh_aabb_list = std::vector<AABB>();
    auto scene_mesh_area_list = std::vector<Float>();
    InitShapesMeshes(mesh_idx_range_list, scene_mesh_aabb_list, scene_mesh_area_list);

    std::cerr << "[info] create shape bvh ..." << std::endl;
    auto meshes_num = scene_mesh_aabb_list.size();
    auto scene_mesh_idx_list = std::vector<uint>(meshes_num);
    for (uint i = 0; i < meshes_num; i++)
        scene_mesh_idx_list[i] = i;

    auto shape_num = mesh_idx_range_list.size();

    bvhnode_list_ = std::vector<BvhNode *>(shape_num);
    shape_info_list = std::vector<BvhNodeInfo>(shape_num);

    shapebvh_list_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&shapebvh_list_, shape_num * sizeof(ShapeBvh)));

    for (uint shape_idx = 0; shape_idx < shape_num; shape_idx++)
    {
        auto scene_shape_mesh_idx_begin = mesh_idx_range_list[shape_idx][0],
             scene_shape_mesh_idx_end = mesh_idx_range_list[shape_idx][1];
        //
        auto mesh_num = scene_shape_mesh_idx_end - scene_shape_mesh_idx_begin;
        auto bvhnode_num = BvhNodeNum(mesh_num);
        auto bvhnode_info_list = std::vector<BvhNodeInfo>(bvhnode_num);
        BuildShapeBvhInfo(0, scene_shape_mesh_idx_begin, scene_shape_mesh_idx_end, scene_mesh_idx_list,
                          scene_mesh_aabb_list, scene_mesh_area_list, bvhnode_info_list);
        shape_info_list[shape_idx] = bvhnode_info_list[0];
        //
        SetMeshesOtherInfo<<<1, 1>>>(scene_shape_mesh_idx_begin, mesh_num, shape_info_list_[shape_idx]->flip_normals,
                                     bvhnode_info_list[0].area, mesh_list_);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());
        //
        BvhNodeInfo *local_bvhnodes_info_list = nullptr;
        CheckCudaErrors(cudaMallocManaged(&local_bvhnodes_info_list, bvhnode_num * sizeof(BvhNodeInfo)));
        cudaMemcpy(local_bvhnodes_info_list, bvhnode_info_list.data(), bvhnode_num * sizeof(BvhNodeInfo),
                   cudaMemcpyHostToDevice);
        //
        BvhNode *bvhnode_list = nullptr;
        CheckCudaErrors(cudaMallocManaged(&bvhnode_list, bvhnode_num * sizeof(BvhNode)));
        auto nx = static_cast<uint>(sqrt(bvhnode_num) + 1);
        auto ny = static_cast<uint>(bvhnode_num / nx + 1);
        auto blocks = dim3(nx / 16 + 1, ny / 16 + 1);
        auto threads = dim3(16, 16);
        CreateShapeBvhNodes<<<blocks, threads>>>(nx, ny, bvhnode_num, mesh_list_, local_bvhnodes_info_list, bvhnode_list);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());

        CreateShapeBvh<<<1, 1>>>(shape_idx, shape_num, bvhnode_list, bvhnode_info_list[0].area, shapebvh_list_);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());

        bvhnode_list_[shape_idx] = bvhnode_list;

        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaFree(local_bvhnodes_info_list));
        local_bvhnodes_info_list = nullptr;
    }
    timer_.PrintTimePassed("create shape bvh ");
}

void Renderer::InitSceneBvh(AABB &scene_aabb)
{
    auto shape_info_list = std::vector<BvhNodeInfo>();
    InitShapeBvh(shape_info_list);

    std::cerr << "[info] create scene bvh ..." << std::endl;

    auto shape_num = shape_info_list.size();
    auto shape_idx_list = std::vector<uint>(shape_num);
    for (uint i = 0; i < shape_num; i++)
        shape_idx_list[i] = i;

    auto node_num = BvhNodeNum(shape_num);
    auto node_info_list = std::vector<BvhNodeInfo>(node_num);
    BuildSceneBvhInfo(0, 0, shape_num, shape_idx_list, shape_info_list, node_info_list);
    scene_aabb = node_info_list[0].aabb;

    BvhNodeInfo *node_info_list_gpu = nullptr;
    CheckCudaErrors(cudaMallocManaged(&node_info_list_gpu, node_num * sizeof(BvhNodeInfo)));
    cudaMemcpy(node_info_list_gpu, node_info_list.data(), node_num * sizeof(BvhNodeInfo), cudaMemcpyHostToDevice);

    scenebvh_node_list_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&scenebvh_node_list_, node_num * sizeof(ShapeBvh)));
    auto nx = static_cast<uint>(sqrt(node_num) + 1);
    auto ny = static_cast<uint>(node_num / nx + 1);
    auto blocks = dim3(nx / 8 + 1, ny / 8 + 1);
    auto threads = dim3(8, 8);
    CreateSceneBvhNodes<<<blocks, threads>>>(nx, ny, node_num, shapebvh_list_, node_info_list_gpu, scenebvh_node_list_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    scenebvh_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&scenebvh_, node_num * sizeof(SceneBvh)));
    CreateSceneBvh<<<1, 1>>>(scenebvh_node_list_, scenebvh_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(node_info_list_gpu));
    node_info_list_gpu = nullptr;
    timer_.PrintTimePassed("create scene bvh ");
}

void Renderer::InitEnvMap(const AABB &scene_aabb)
{
    if (!env_map_info_)
        return;

    if (env_map_info_->radiance_idx == kUintMax)
        return;

    auto radius = myvec::length(scene_aabb.max() - scene_aabb.min()) * 0.5;
    auto pdf_area = 1.0 / (4.0 * kPi * radius * radius);

    gmat4 *to_local = nullptr;
    if (env_map_info_->to_local)
    {
        CheckCudaErrors(cudaMallocManaged(&to_local, sizeof(gmat4)));
        cudaMemcpy(to_local, env_map_info_->to_local, sizeof(gmat4), cudaMemcpyHostToDevice);
    }

    env_map_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&env_map_, sizeof(EnvMap)));
    CreateEnvMap<<<1, 1>>>(pdf_area, to_local, env_map_info_->radiance_idx, texture_list_, env_map_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(to_local));
    to_local = nullptr;
}

void Renderer::initIntegratorCamera()
{
    auto emitter_num = emitter_shape_idx_list_.size();
    emitter_idx_list_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&emitter_idx_list_, emitter_num * sizeof(uint)));
    cudaMemcpy(emitter_idx_list_, emitter_shape_idx_list_.data(), emitter_num * sizeof(uint), cudaMemcpyHostToDevice);

    integrator_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&integrator_, sizeof(Integrator)));
    InitIntegrator<<<1, 1>>>(integrator_info_, scenebvh_, shapebvh_list_, emitter_idx_list_, emitter_num, env_map_, integrator_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    camera_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&camera_, sizeof(Camera)));
    InitCamera<<<1, 1>>>(camera_info_, camera_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    d_rand_state_ = nullptr;
    auto resolution = camera_info_.height * camera_info_.width;
    CheckCudaErrors(cudaMalloc((void **)&d_rand_state_, resolution * sizeof(curandState)));
    auto tx = camera_info_.width > 8 ? 8 : camera_info_.width;
    auto ty = camera_info_.height > 8 ? 8 : camera_info_.height;
    auto nx = camera_info_.width;
    auto ny = camera_info_.height;
    auto blocks = dim3(nx / tx + 1, ny / ty + 1);
    auto threads = dim3(tx, ty);
    RenderInit<<<blocks, threads>>>(nx, ny, resolution, d_rand_state_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
}

void Renderer::Render(const std::string &output_filename)
{
    timer_.Reset();
    auto scene_aabb = AABB();
    InitSceneBvh(scene_aabb);
    InitEnvMap(scene_aabb);
    initIntegratorCamera();
    timer_.PrintTimePassed("prepare work");

    auto resolution = camera_info_.height * camera_info_.width;
    float *frame_data = nullptr;
    CheckCudaErrors(cudaMallocManaged((void **)&frame_data, 3 * resolution * sizeof(float)));
    auto tx = camera_info_.width > 16 ? 16 : camera_info_.width;
    auto ty = camera_info_.height > 16 ? 16 : camera_info_.height;
    auto nx = camera_info_.width;
    auto ny = camera_info_.height;
    auto blocks = dim3(nx / tx + 1, ny / ty + 1);
    auto threads = dim3(tx, ty);

#ifdef PRINT_PROGRESS
    volatile int *device_progress, *host_progess;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    CheckCudaErrors(cudaGetLastError());
    cudaHostAlloc((void **)&host_progess, sizeof(int), cudaHostAllocMapped);
    CheckCudaErrors(cudaGetLastError());
    cudaHostGetDevicePointer((int **)&device_progress, (int *)host_progess, 0);
    CheckCudaErrors(cudaGetLastError());
    *host_progess = 0;

    unsigned int num_blocks = blocks.x * blocks.y;

    Timer timer2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    RenderProcess<<<blocks, threads>>>(nx, ny, resolution, d_rand_state_, camera_, integrator_, frame_data, device_progress);
    cudaEventRecord(stop);
    float kernel_progress = 0.0f;
    while (kernel_progress < 0.9999)
    {
        cudaEventQuery(stop);
        kernel_progress = static_cast<float>(*host_progess) / num_blocks;
        timer2.PrintProgress(kernel_progress);
    }
    cudaEventSynchronize(stop);
#else
    Timer timer2;
    RenderProcess<<<blocks, threads>>>(nx, ny, resolution, d_rand_state_, camera_, integrator_, frame_data);
#endif
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    timer2.PrintTimePassed2("rendering");
    timer_.PrintTimePassed("rendering");

    auto frame = Frame();
    frame.width = camera_info_.width;
    frame.height = camera_info_.height;
    frame.data = std::vector<float>(3 * resolution);

    cudaMemcpy(&(frame.data[0]), frame_data, 3 * resolution * sizeof(float), cudaMemcpyDeviceToHost);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(frame_data));
    frame_data = nullptr;
    WriteImage(frame, output_filename);

    timer_.PrintTimePassed("all the work");
}