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

    Point epsHit = hit + epsilon * shadingN;

    // Add diffuse and specular components.
    for (auto const &light : lights)
    {
        Vector L = (light->position - hit).normalized();

        //-------------------------------------------------------------------MYOWN PART START
        Ray shadowRay(epsHit, L);
        pair<ObjectPtr, Hit> shadowHit = castRay(shadowRay);
        if (renderShadows && shadowHit.first && shadowHit.second.t < (light->position - hit).length() ){
            continue;
        }

        //-------------------------------------------------------------------MYOWN PART END

        // Add diffuse.
        double diffuse = std::max(shadingN.dot(L), 0.0);
        color += diffuse * material.kd * light->color * matColor;

        // Add specular.
        Vector reflectDir = reflect(-L, shadingN);
        double specAngle = std::max(reflectDir.dot(V), 0.0);
        double specular = std::pow(specAngle, material.n);

        color += specular * material.ks * light->color;
    }

    if (depth > 0 and material.isTransparent)
    {
        // The object is transparent, and thus refracts and reflects light.
        // Use Schlick's approximation to determine the ratio between the two.

        double ni, nt;
        Vector D = ray.D;

        if(N.dot(V) <= 0.0){
            N = -N;
            ni = material.nt;
            nt = 1;
        } else {
            ni = 1;
            nt = material.nt;
        }
        double r0 = ((ni - nt)/(ni + nt))*((ni - nt)/(ni + nt));
        double incidentAngle = acos(N.dot(V));
        if (incidentAngle > 3.1416 / 2) printf("angle %lf\n", incidentAngle);
        double refractAngle = asin( ni * sin(incidentAngle) / nt );
        double kr = r0 + (1-r0)*pow((1-cos(incidentAngle)),5);
        double kt = 1 - kr;

        //printf("%lf %lf\n", incidentAngle/3.1415*180, refractAngle/3.1415*180);
        //Vector R = r0 + (1-r0)*(1-cos())

        Vector B = ( N*(N.dot(V)) + D) / sin(incidentAngle);

        Vector T = B*sin(refractAngle) - N*cos(refractAngle);
        //printf("%lf\n", N.dot(B));
        //printf("D %lf %lf %lf\n T %lf %lf %lf\n", D.x, D.y, D.z, T.x, T.y, T.z);

        Vector reflecDir = reflect(-V, N);
        Ray reflecRay = Ray(hit + epsilon*N, reflecDir);
        Ray refracRay = Ray(hit - epsilon*N, T);

        //if(kr > 1 || kt > 1) printf("%lf %lf\n", kr, kt);
        Color reflecColor = trace(reflecRay, depth-1);
        color += kr * reflecColor;

        Color refracColor = trace(refracRay, depth-1);
        color += kt * refracColor;

    }
    else if (depth > 0 and material.ks > 0.0)
    {
        // The object is not transparent, but opaque.
        Vector reflecDir = reflect(-V, N);
        Ray reflecRay = Ray(epsHit, reflecDir);

        Color colorAdd = trace(reflecRay, depth-1);
        color += material.ks * colorAdd;
    }

    return color;
}

void Scene::render(Image &img)
{
	Color col;
	double step;
	unsigned w = img.width();
    unsigned h = img.height();
	int i, j, ssf = (int)supersamplingFactor;

    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x)
        {
			col = Color(0, 0, 0);
            step = 0.5 / supersamplingFactor;
            for (i = -1; i < ssf * 2 + 1; i += 2)
            {
                for (j = -1; j < ssf * 2 + 1; j += 2)
                {
                    Point pixel(x + step * i, h - 1 - y + step * j, 0);
                    Ray ray(eye, (pixel - eye).normalized());
                    col += trace(ray, recursionDepth).operator/(pow(supersamplingFactor + 1, 2));
                }
            }

            col.clamp();
            img(x, y) = col;
        }
}

// --- Misc functions ----------------------------------------------------------

// Defaults
Scene::Scene()
:
    objects(),
    lights(),
    eye(),
    renderShadows(false),
    recursionDepth(0),
    supersamplingFactor(1)
{}

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
