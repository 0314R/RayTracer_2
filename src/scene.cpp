#include "scene.h"

#include "hit.h"
#include "image.h"
#include "material.h"
#include "ray.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

pair<ObjectPtr, Hit> Scene::castRay(Ray const &ray) const
{
    // Find hit object and distance
    Hit min_hit(numeric_limits<double>::infinity(), Vector());
    ObjectPtr obj = nullptr;
    for (unsigned idx = 0; idx != objects.size(); ++idx)
    {
        Hit hit(objects[idx]->intersect(ray));
        if (hit.t < min_hit.t)
        {
            min_hit = hit;
            obj = objects[idx];
        }
    }

    return pair<ObjectPtr, Hit>(obj, min_hit);
}

Color Scene::trace(Ray const &ray, unsigned depth)
{
    pair<ObjectPtr, Hit> mainhit = castRay(ray);
    ObjectPtr obj = mainhit.first;
    Hit min_hit = mainhit.second;

    // No hit? Return background color.
    if (!obj)
        return Color(0.0, 0.0, 0.0);

    Material const &material = obj->material;
    Point hit = ray.at(min_hit.t);
    Vector V = -ray.D;

    // Pre-condition: For closed objects, N points outwards.
    Vector N = min_hit.N;

    // The shading normal always points in the direction of the view,
    // as required by the Phong illumination model.
    Vector shadingN;
    if (N.dot(V) >= 0.0)
        shadingN = N;
    else
        shadingN = -N;

    Color matColor = material.color;

    // Add ambient once, regardless of the number of lights.
    Color color = material.ka * matColor;

    Point epsilonH = hit + epsilon * shadingN;
    Vector uv = obj->toUV(hit);

    if (material.hasTexture)
    {
        color = material.ka * material.texture.colorAt(uv.x, 1 - uv.y);
    }

    // Add diffuse and specular components.
    for (auto const &light : lights)
    {
        Vector L = (light->position - hit).normalized();

        Ray shadowRay(epsilonH, L);
        pair<ObjectPtr, Hit> shadowH = castRay(shadowRay);
        if (renderShadows && shadowH.first && shadowH.second.t < (light->position - hit).length())
        {
            continue;
        }

        // Add diffuse.
        double dotNormal = shadingN.dot(L);
        double diffuse = std::max(dotNormal, 0.0);
        if (material.hasTexture)
        {
            Vector uv = obj->toUV(hit);
            Color texColor = material.texture.colorAt(uv.x, 1.0 - uv.y);
            color += diffuse * material.kd * light->color * texColor;
        }
        else
        {
            color += diffuse * material.kd * light->color * matColor;
        }

        // Add specular.
        if (dotNormal > 0)
        {
            Vector reflectDir = reflect(-L, shadingN); // Note: reflect(..) is not given in the framework.
            double specAngle = std::max(reflectDir.dot(V), 0.0);
            double specular = std::pow(specAngle, material.n);

            color += specular * material.ks * light->color;
        }
    }

    if (depth > 0 and material.isTransparent)
    {
        // The object is transparent, and thus refracts and reflects light.
        // Use Schlick's approximation to determine the ratio between the two.
        double ni, nt;
        Vector direction = ray.D;

        if (N.dot(V) > 0.0)
        {
            ni = 1;
            nt = material.nt;
        }
        else
        {
            N = -N;
            ni = material.nt;
            nt = 1;
        }

        double kr0 = ((ni - nt) / (ni + nt)) * ((ni - nt) / (ni + nt));
        double incident = acos(N.dot(V));
        double refractive = asin(ni * sin(incident) / nt);
        double kr = kr0 + (1 - kr0) * pow((1 - cos(incident)), 5);
        double kt = 1 - kr;

        Vector B = (N * (N.dot(V)) + direction) / sin(incident);
        Vector T = B * sin(refractive) - N * cos(refractive);
        Vector reflectDir = reflect(-V, N);

        Ray reflectRay = Ray(hit + epsilon * N, reflectDir);
        Ray refractRay = Ray(hit - epsilon * N, T);

        Color reflectColor = trace(reflectRay, depth - 1);
        color += kr * reflectColor;

        Color refractColor = trace(refractRay, depth - 1);
        color += kt * refractColor;
    }
    else if (depth > 0 and material.ks > 0.0)
    {
        // The object is not transparent, but opaque.
        Vector directionR = reflect(-V, N);
        Ray rayReflect = Ray(epsilonH, directionR);

        Color newColor = trace(rayReflect, depth - 1);
        color += material.ks * newColor;
    }

    return color;
}

void Scene::render(Image &img)
{
    unsigned w = img.width();
    unsigned h = img.height();

    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x)
        {
            // Color col = Color(0, 0, 0);
            // double step = 0.5 / supersamplingFactor;
            // for (int z = -1; z < (int)supersamplingFactor * 2 + 1; z += 2)
            // {
            //     for (int a = -1; a < (int)supersamplingFactor * 2 + 1; a += 2)
            //     {
            //         Point pixel(x + step * z, h - 1 - y + step * a, 0);
            //         Ray ray(eye, (pixel - eye).normalized());
            //         col += trace(ray, recursionDepth).operator/(pow(supersamplingFactor + 1, 2));
            //     }
            // }

            Point pixel(x + 0.5, h - 1 - y + 0.5, 0);
            Ray ray(eye, (pixel - eye).normalized());
            Color col = trace(ray, recursionDepth);
            col.clamp();
            img(x, y) = col;
        }
}

// --- Misc functions ----------------------------------------------------------

// Defaults
Scene::Scene()
    : objects(),
      lights(),
      eye(),
      renderShadows(false),
      recursionDepth(0),
      supersamplingFactor(1)
{
}

void Scene::addObject(ObjectPtr obj)
{
    objects.push_back(obj);
}

void Scene::addLight(Light const &light)
{
    lights.push_back(LightPtr(new Light(light)));
}

void Scene::setEye(Triple const &position)
{
    eye = position;
}

unsigned Scene::getNumObject()
{
    return objects.size();
}

unsigned Scene::getNumLights()
{
    return lights.size();
}

void Scene::setRenderShadows(bool shadows)
{
    renderShadows = shadows;
}

void Scene::setRecursionDepth(unsigned depth)
{
    recursionDepth = depth;
}

void Scene::setSuperSample(unsigned factor)
{
    supersamplingFactor = factor;
}
