/*
  bvh.h - A single header implementation of shallow bounding volume 
  hierarchies [1] using spatial splits [2].

  LICENSE:
    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or
    distribute this software, either in source code form or as a compiled
    binary, for any purpose, commercial or non-commercial, and by any
    means.

    In jurisdictions that recognize copyright laws, the author or authors
    of this software dedicate any and all copyright interest in the
    software to the public domain. We make this dedication for the benefit
    of the public at large and to the detriment of our heirs and
    successors. We intend this dedication to be an overt act of
    relinquishment in perpetuity of all present and future rights to this
    software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.

    For more information, please refer to <http://unlicense.org/>

  HISTORY:
    0.1   Initial release.

  REFERENCES:
    [1] Dammertz, Holger, Johannes Hanika, and Alexander Keller. "Shallow
        bounding volume hierarchies for fast SIMD ray tracing of incoherent 
        rays." Computer Graphics Forum. Vol. 27. No. 4. Blackwell Publishing 
        Ltd, 2008.
    [2] Stich, Martin, Heiko Friedrich, and Andreas Dietrich. "Spatial splits 
        in bounding volume hierarchies." Proceedings of the Conference on High
        Performance Graphics 2009. ACM, 2009.
*/
#ifndef BVH_H
#define BVH_H

#include <stdbool.h>
#include <stdint.h>

#define BVH_VERSION "0.1.0"

typedef struct bvh__Data *      bvh_Handle;

bvh_Handle  (bvh_build) (const uint32_t vertexids[], const float vertices[],
                         uint32_t ntriangles);
bool        (bvh_trace) (const bvh_Handle bvh, float *t, float *u, float *v,
                         uint32_t *faceid, const float o[3], const float d[3],
                         float tmin, float tmax);
void        (bvh_free)  (bvh_Handle *bvh);

#endif

#ifdef BVH_IMPLEMENTATION

#include <stdlib.h>
#include <setjmp.h>
#include <memory.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <xmmintrin.h>

/* -- Configuration --------------------------------------------------------- */

#ifndef BVH_MAX_TRIANGLES
# define BVH_MAX_TRIANGLES        16     /* Max. # of triangles a node holds. */
#else
# if BVH_MAX_TRIANGLES < 4 || BVH_MAX_TRIANGLES > 64 || BVH_MAX_TRIANGLES % 4
#   error `BVH_MAX_TRIANGLES' needs to be a multiple of four and in between 4 and 64.
# endif
#endif

#ifndef BVH_SAH_COST
# define BVH_SAH_COST             0.025f
#endif

#ifndef BVH_SAH_NUM_BUCKETS
# define BVH_SAH_NUM_BUCKETS      12
#endif

#ifndef BVH_SPATIAL_SPLIT_ALPHA
# define BVH_SPATIAL_SPLIT_ALPHA          0.01f
#endif

/* -- Misc. utilities ------------------------------------------------------- */

#define BVH__EPS                0.0001f
#define BVH__MEPSILON           (0.5f * FLT_EPSILON)       /* Machine epsilon */

#define bvh__assert(x)          assert(x)
#define bvh__ispow2(x)          (!((x) & ((x) - 1)) && (x))
#define bvh__min(a, b)          ((a) < (b) ? (a) : (b))
#define bvh__max(a, b)          ((a) > (b) ? (a) : (b))
#define bvh__clamp(x, a, b)     (bvh__max(a, bvh__min(x, b)))
#define bvh__almeq(a, b)        (fabs((a) - (b)) < BVH__EPS)
#define bvh__swap(T, a, b)      do { T c = (a); (a) = (b); (b) = (c); } while (0)
#define bvh__inrange(i, a, b)   ((i) >= (a) && (i) <= (b))
#define bvh__ceildiv(a, b)      ((a + (b - 1)) / b)
#define bvh__prevmult4(n)       (((n) + 0) & ~0x03)
#define bvh__nextmult4(n)       (((n) + 3) & ~0x03)
#define bvh__closemult4(n)      (((n) + 2) & ~0x03)

typedef float bvh__Vec3[3];

static void bvh__sub3(float c[3], const float a[3], const float b[3])
{
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
}

typedef struct {
  bvh__Vec3 b[2];
} bvh__BBox;

static void bvh__bbox_center(bvh__Vec3 c, const bvh__BBox *bbox)
{
  c[0] = 0.5f * (bbox->b[0][0] + bbox->b[1][0]);
  c[1] = 0.5f * (bbox->b[0][1] + bbox->b[1][1]);
  c[2] = 0.5f * (bbox->b[0][2] + bbox->b[1][2]);
}

static void bvh__bbox_extend(bvh__BBox *b, const float v[3])
{
  b->b[0][0] = bvh__min(b->b[0][0], v[0]);
  b->b[0][1] = bvh__min(b->b[0][1], v[1]);
  b->b[0][2] = bvh__min(b->b[0][2], v[2]);
  b->b[1][0] = bvh__max(b->b[1][0], v[0]);
  b->b[1][1] = bvh__max(b->b[1][1], v[1]);
  b->b[1][2] = bvh__max(b->b[1][2], v[2]);
}

static void bvh__bbox_union(bvh__BBox *a, const bvh__BBox *b)
{
  a->b[0][0] = bvh__min(a->b[0][0], b->b[0][0]);
  a->b[0][1] = bvh__min(a->b[0][1], b->b[0][1]);
  a->b[0][2] = bvh__min(a->b[0][2], b->b[0][2]);
  a->b[1][0] = bvh__max(a->b[1][0], b->b[1][0]);
  a->b[1][1] = bvh__max(a->b[1][1], b->b[1][1]);
  a->b[1][2] = bvh__max(a->b[1][2], b->b[1][2]);
}

static void bvh__bbox_intersect(bvh__BBox *a, const bvh__BBox *b)
{
  a->b[0][0] = bvh__max(a->b[0][0], b->b[0][0]);
  a->b[0][1] = bvh__max(a->b[0][1], b->b[0][1]);
  a->b[0][2] = bvh__max(a->b[0][2], b->b[0][2]);
  a->b[1][0] = bvh__min(a->b[1][0], b->b[1][0]);
  a->b[1][1] = bvh__min(a->b[1][1], b->b[1][1]);
  a->b[1][2] = bvh__min(a->b[1][2], b->b[1][2]);
}

static float bvh__bbox_area(const bvh__BBox *box)
{
  float d[3];
  if (box->b[0][0] > box->b[1][0] ||
      box->b[0][1] > box->b[1][1] ||
      box->b[0][2] > box->b[1][2])
    return 0.0f;
  bvh__sub3(d, box->b[1], box->b[0]);
  return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
}

static bvh__BBox bvh__bbox_empty()
{
  const bvh__BBox b = {
    .b[0] = {+FLT_MAX, +FLT_MAX, +FLT_MAX},
    .b[1] = {-FLT_MAX, -FLT_MAX, -FLT_MAX}
  };
  return b;
}

static void bvh__bbox4point(bvh__BBox *b, const float x[3])
{
  memcpy(b->b[0], x, sizeof(b->b[0]));
  memcpy(b->b[1], x, sizeof(b->b[1]));
}

static void bvh__bbox4triangle(bvh__BBox *bbox, const float v0[3],
                               const float v1[3], const float v2[3])
{
  bbox->b[0][0] = bvh__min(v0[0], bvh__min(v1[0], v2[0]));
  bbox->b[0][1] = bvh__min(v0[1], bvh__min(v1[1], v2[1]));
  bbox->b[0][2] = bvh__min(v0[2], bvh__min(v1[2], v2[2]));
  bbox->b[1][0] = bvh__max(v0[0], bvh__max(v1[0], v2[0]));
  bbox->b[1][1] = bvh__max(v0[1], bvh__max(v1[1], v2[1]));
  bbox->b[1][2] = bvh__max(v0[2], bvh__max(v1[2], v2[2]));
}

typedef struct {
  __m128 bx[2];
  __m128 by[2];
  __m128 bz[2];
} bvh__QBBox;

static void bvh__qbbox_set(bvh__QBBox *quad, uint32_t i,
                           const bvh__BBox *bbox)
{
  float *boxes = (float *)quad;
  bvh__assert(bvh__inrange(i, 0, 3));
  boxes[i +  0] = bbox->b[0][0];
  boxes[i +  4] = bbox->b[1][0];
  boxes[i +  8] = bbox->b[0][1];
  boxes[i + 12] = bbox->b[1][1];
  boxes[i + 16] = bbox->b[0][2];
  boxes[i + 20] = bbox->b[1][2];
}

static void bvh__qbbox_setzero(bvh__QBBox *quad, uint32_t i)
{
  float *boxes = (float *)quad;
  bvh__assert(bvh__inrange(i, 0, 3));
  boxes[i +  0] = 0.0f;
  boxes[i +  4] = 0.0f;
  boxes[i +  8] = 0.0f;
  boxes[i + 12] = 0.0f;
  boxes[i + 16] = 0.0f;
  boxes[i + 20] = 0.0f;
}

typedef uint32_t bvh_id_quad_t[4]; /* Ids for a triangle bundle.    */
typedef struct bvh_triangle_quad {
  __m128 v0x, v1x, v2x;
  __m128 v0y, v1y, v2y;
  __m128 v0z, v1z, v2z;
} bvh__QTriangle;

static void bvh__qtriangle_set(bvh__QTriangle *quad,
                               uint32_t i,
                               const float v0[3],
                               const float v1[3],
                               const float v2[3])
{
  float *triangles = (float *)quad;
  bvh__assert(bvh__inrange(i, 0, 3));
  triangles[i +  0] = v0[0];
  triangles[i +  4] = v1[0];
  triangles[i +  8] = v2[0];
  triangles[i + 12] = v0[1];
  triangles[i + 16] = v1[1];
  triangles[i + 20] = v2[1];
  triangles[i + 24] = v0[2];
  triangles[i + 28] = v1[2];
  triangles[i + 32] = v2[2];
}
/* -------------------------------------------------------------------------- */

#define BVH__NODE_EMPTY         UINT32_MAX

#define bvh_isempty(id)                                                        \
  ((id) == BVH__NODE_EMPTY)
#define bvh_isleaf(id)                                                         \
  ((id) & (1 << 31))

/* Encodes a 32 bit integer with leaf node data. The most significant bit
   identifies the node as leaf, the four following bits hold information about
   the number of triangle bundles which is followed by a reference into the
   bundle data of the bvh (27 bits). */
#define bvh__leaf_encodeid(s, n)                                               \
  ((1 << 31) | ((((n) - 1) << 27) & 0x78000000) | ((s) & 0x07FFFFFF))
#define bvh__leaf_nquads(id)                                                   \
  (((((id) & 0x78000000) >> 27) & 0x000000FF) + 1)
#define bvh__leaf_quadid(id)                                                   \
  ((id) & 0x07FFFFFF)

typedef struct {
  bvh__QBBox childbboxes;
  /* References to the four child nodes. Each reference also encodes the
     type of the child node: if the reference is 'BVH__NODE_EMPTY', the child
     does not exist, else if bvh_isleaf() evaluates to true given the reference,
     the child is a leaf node, otherwise the child is an inner node. */
  uint32_t childids[4];
  uint32_t axis[3];
  uint32_t padding;
} bvh__Node;

typedef struct bvh__Data {
  bvh__BBox bounds;
  bvh__Node *nodes; /* 128 bit aligned. */
  uint32_t nnodes;
  uint32_t root;
  bvh__QTriangle *trianglequads; /* Triangle quadruples */
  bvh_id_quad_t *idquads; /* (Input) triangle ids quadruples */
  uint32_t nquads;
} bvh__Data;

/* -- Construction ---------------------------------------------------------- */

typedef struct {
  uint32_t count;
  bvh__BBox bbox;
} bvh__SAHBucket;

/* Represents a set of triangles, which is one result of a splitting operation.
   Triangles can be chopped during splitting splitting (see [3] 4.2.), hence
   the 'bbox' of the set does not need to be equal to the bounding box of all
   referenced triangles. */
typedef struct {
  bvh__BBox bbox; /* (Chopped) bbox */
  uint32_t start;
  uint32_t end;
} bvh__Set;

#define bvh__set_isempty(set) ((set)->start >= (set)->end)

typedef struct {
  jmp_buf exc;

  bvh__Data *bvh;
  float boxarea;        /* Area of the bvh's bounding box. */

  /* Build input. */
  const uint32_t *vertexids;
  const float *vertices;
  uint32_t ntriangles;

  /* Build info */
  bvh__BBox *bboxes;
  bvh__Vec3 *centroids;

  /* Triangle id stack */
  uint32_t *ids;
  uint32_t nids;
  uint32_t idcap;

  /* Bvh meta data */
  uint32_t nodescap;
  uint32_t quadscap;

  uint32_t splitaxis; /* Splitaxis for the current triangle set. */
  float cmin;         /* cbbox.b[0][splitaxis] */
  float cmax;         /* cbbox.b[1][splitaxis] */
  bvh__BBox cbbox;    /* BBox for the centroids of the current triangle set. */
  bvh__Set seta;      /* Split set A. */
  bvh__Set setb;      /* Split set B. */
  float cost;         /* SAH cost for the split. */

  bvh__SAHBucket buckets[BVH_SAH_NUM_BUCKETS];  /* SAH buckets. */
} bvh__BuildState;

#define BVH__TRY(bs)      do { if (setjmp((bs)->exc) == 0) {
#define BVH__CATCH(bs)    } else {
#define BVH__END(bs)      } } while (0);
#define BVH__THROW(bs)    longjmp((bs)->exc, 1);

typedef struct {
  void *start;
  uint32_t size;
} bvh__MemHeader;

static void *bvh__amalloc(bvh__BuildState *bs, uint32_t sz, uint32_t al)
{
  bvh__MemHeader *h;
  uint8_t *m, *ptr = NULL;
  bvh__assert(bvh__ispow2(al));
  if (!sz) return NULL;
  m = malloc(sz + al + sizeof(*h));
  if (m) {
    ptr = (uint8_t *)((uintptr_t)(m + sizeof(*h) + al) & ~((uintptr_t)al - 1));
    h = &((bvh__MemHeader *)ptr)[-1];
    h->start = m;
    h->size = sz;
  } else
    BVH__THROW(bs)
  return ptr;
}

static void bvh__afree(void *ptr)
{
  bvh__MemHeader *h = NULL;
  if (ptr) {
    h = &((bvh__MemHeader *)ptr)[-1];
    free(h->start);
  }
}

static void *bvh__arealloc(bvh__BuildState *bs, void *ptr,
                           uint32_t sz, uint32_t al) 
{
  void *nptr = bvh__amalloc(bs, sz, al);
  if (nptr && ptr) {
    bvh__MemHeader *h;
    uint32_t nsz;
    h = &((bvh__MemHeader *)ptr)[-1];
    nsz = sz > h->size ? h->size : sz;
    memcpy(nptr, ptr, nsz);
  }
  return nptr;
}

/* Pushes 'n' new elements on the id stack. Returns the previous top index
   of the stack. */
static uint32_t bvh__ids_push(bvh__BuildState *bs, uint32_t n)
{
  uint32_t cap = bs->idcap ? bs->idcap : 1;
  while (bs->nids + n > cap) cap *= 2;
  if (cap > bs->idcap) {
    bs->ids = (uint32_t *)bvh__arealloc(bs, bs->ids, cap * sizeof(uint32_t),
                                            sizeof(uint32_t));
    bs->idcap = cap;
  }
  bs->nids += n;
  return bs->nids - n;
}

typedef struct {
  uint32_t splitaxis;
  bvh__Set seta;
  bvh__Set setb;
} bvh__SplitResult;

static bvh__SplitResult bvh__splitres(const bvh__BuildState *bs)
{
  bvh__SplitResult rv = {bs->splitaxis, bs->seta, bs->setb};
  return rv;
}

static void bvh__set4range(bvh__BuildState *bs, bvh__Set *set,
                           uint32_t s, uint32_t e)
{
  uint32_t i;
  set->start = s;
  set->end = e;
  set->bbox = bvh__bbox_empty();
  for (i = s; i < e; ++i)
    bvh__bbox_union(&set->bbox, &bs->bboxes[bs->ids[i]]);
}

/* -- Default triangle set splitting ---------------------------------------- */

static uint32_t bvh__sah_bucketid(const bvh__BuildState *bs, float x)
{
  return bvh__clamp((x - bs->cmin) / (bs->cmax - bs->cmin) * BVH_SAH_NUM_BUCKETS,
                     0, BVH_SAH_NUM_BUCKETS - 1);
}

static float bvh__sah_cost(uint32_t na, const bvh__BBox *ba,
                           uint32_t nb, const bvh__BBox *bb,
                           const bvh__BBox *bn)
{
  return (bvh__nextmult4(na) * bvh__bbox_area(ba) + 
          bvh__nextmult4(nb) * bvh__bbox_area(bb)) / 
          bvh__bbox_area(bn) + BVH_SAH_COST;
}

static void bvh__init_sah_buckets(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t i = 0;
  for (i = 0; i < BVH_SAH_NUM_BUCKETS; ++i) {
    bs->buckets[i].bbox = bvh__bbox_empty();
    bs->buckets[i].count = 0;
  }
  for (i = set->start; i < set->end; ++i) {
    float c = bs->centroids[bs->ids[i]][bs->splitaxis];
    uint32_t bid = bvh__sah_bucketid(bs, c);
    bvh__bbox_union(&bs->buckets[bid].bbox, &bs->bboxes[bs->ids[i]]);
    bs->buckets[bid].count++;
  }
  /* Intersect each bucket bbox with the split sets bbox, since the
     split set my contain chopped triangles. */
  for (i = 0; i < BVH_SAH_NUM_BUCKETS; ++i)
    bvh__bbox_intersect(&bs->buckets[i].bbox, &set->bbox);
}

static bool bvh__split_default(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t i = 0;
  float curcost = FLT_MAX;
  uint32_t costminid = -1;
  bvh__assert(set->start < (set->end - 1)); /* Require at least two triangles. */
  bs->cost = FLT_MAX;
  bvh__init_sah_buckets(bs, set);
  for (i = 0; i < BVH_SAH_NUM_BUCKETS - 1; ++i) {
    /* Get the id of the bucket *after* which the SAH split cost is minimal. */
    uint32_t j;
    uint32_t na = 0, nb = 0;
    bvh__BBox ba = bvh__bbox_empty();
    bvh__BBox bb = bvh__bbox_empty();
    for (j = 0; j <= i; ++j) {
      bvh__bbox_union(&ba, &bs->buckets[j].bbox);
      na += bs->buckets[j].count;
    }
    for (j = i + 1; j < BVH_SAH_NUM_BUCKETS; ++j) {
      bvh__bbox_union(&bb, &bs->buckets[j].bbox);
      nb += bs->buckets[j].count;
    }
    curcost = bvh__sah_cost(na, &ba, nb, &bb, &set->bbox);
    if (curcost < bs->cost) {
      /* Keep track of the data related to the bucket after which the
         splitting cost is minimal. */
      costminid = i;
      bs->cost = curcost;
      bs->seta.bbox = ba;
      bs->setb.bbox = bb;
    }
  }
  if (bs->cost >= (float)(set->end - set->start)) {
    /* Handle the case, when it is not worth splitting the set. */
    bs->cost = (float)(set->end - set->start);
    bs->seta = *set;
    memset(&bs->setb, 0, sizeof(bs->setb));
    return false;
  } else {
    /* Partition triangles. */
    uint32_t start = set->start;
    uint32_t end = set->end - 1;
    while (start < end) {
      uint32_t *ids, *ide;
      uint32_t bids, bide;
      float axs, axe;
      ids = &bs->ids[start];
      ide = &bs->ids[end];
      axs = bs->centroids[*ids][bs->splitaxis]; /* NOTE: Centroid may lie out of 
                                                         `set->bbox', when spatial 
                                                         splitting is used. */
      axe = bs->centroids[*ide][bs->splitaxis];
      bids = bvh__sah_bucketid(bs, axs);
      bide = bvh__sah_bucketid(bs, axe);
      if (bids > costminid && bide <= costminid)
        bvh__swap(uint32_t, *ids, *ide);
      start = bids <= costminid ? start + 1 : start;
      end = bide > costminid ? end - 1 : end;
    }
    /* Set 'start' to end of the first set. */
    {
      float c = bs->centroids[bs->ids[start]][bs->splitaxis];
      uint32_t bid = bvh__sah_bucketid(bs, c);
      start =  bid > costminid ? start : start + 1;
    }
    bs->seta.start = set->start;
    bs->seta.end = start;
    bs->setb.start = start;
    bs->setb.end = set->end;
    return true;
  }
}

/* -- Revising the default split -------------------------------------------- */

#define bvh__rev_gt(bs, ida, idb)                                              \
  ((bs)->bboxes[ida].b[1][(bs)->splitaxis] >                                   \
   (bs)->bboxes[idb].b[1][(bs)->splitaxis])

static void bvh__rev_sort(bvh__BuildState *bs, uint32_t s, uint32_t e)
{
  uint32_t i;
  if (bvh__rev_gt(bs, bs->ids[s], bs->ids[s + 1]))
    bvh__swap(uint32_t, bs->ids[s], bs->ids[s + 1]);
  for (i = s + 2; i < e; ++i) {
    if (bvh__rev_gt(bs, bs->ids[s], bs->ids[i])) {
      bvh__swap(uint32_t, bs->ids[s], bs->ids[s + 1]);
      bvh__swap(uint32_t, bs->ids[s], bs->ids[i]);
    } else if (bvh__rev_gt(bs, bs->ids[s + 1], bs->ids[i]))
      bvh__swap(uint32_t, bs->ids[s + 1], bs->ids[i]);
  }
}

static int32_t bvh__adjustmid(int32_t s, int32_t e, int32_t m)
{
  int32_t m0, m1;
  m0 = s + bvh__nextmult4(m - s);
  m1 = e - bvh__prevmult4(e - m);
  bvh__assert(m0 > m && m1 > m);
  if (m0 - m > m1 - m)
    return m1;
  else
    return m0;
}

static void bvh__updatesets(bvh__BuildState *bs, const bvh__Set *set, uint32_t m)
{
  int i = 0;
  float cost;
  bvh__BBox bba, bbb;
  if (m >= bs->setb.end || m - bs->seta.end >= 3)
    return;
  bvh__rev_sort(bs, bs->setb.start, bs->setb.end);
  bba = bs->seta.bbox;
  for (i = bs->seta.end; i < m; ++i)
    bvh__bbox_union(&bba, &bs->bboxes[bs->ids[i]]);
  bbb = bvh__bbox_empty();
  for (i = m; i < bs->setb.end; ++i)
    bvh__bbox_union(&bbb, &bs->bboxes[bs->ids[i]]);
  cost = bvh__sah_cost(m - bs->seta.start, &bba, bs->setb.end - m, &bbb,
                       &set->bbox);
  if (cost < bs->cost) {
    bs->cost = cost;
    bs->seta.bbox = bba;
    bs->setb.bbox = bbb;
    bs->seta.end = m;
    bs->setb.start = m;
  }
}

/* This function tries to adjust the two split sets s.t. at least one set has 
   an amount of triangles that is a multiple of four. */
static void bvh__revisesplit(bvh__BuildState *bs, const bvh__Set *set)
{
  int32_t m;
  bvh__assert(bs->seta.end == bs->setb.start);
  bvh__assert(bs->setb.end - bs->seta.start >= 4);
  if (!((bs->seta.end - bs->seta.start) % 4) ||
      !((bs->setb.end - bs->setb.start) % 4))
    return;
  m = bvh__adjustmid(bs->seta.start, bs->setb.end, bs->setb.start);
  bvh__assert(m > bs->seta.end);
  bvh__updatesets(bs, set, m);
}

/* -- Half splits ----------------------------------------------------------- */

#define bvh__qselect_val(bs, i)                                                \
  ((bs)->centroids[(bs)->ids[(i)]][(bs)->splitaxis])

static int bvh__qselect_partition(bvh__BuildState *bs, uint32_t s, uint32_t e)
{
  float pivot = bvh__qselect_val(bs, e - 1);
  uint32_t j = s, i;
  for (i = s; i < e - 1; ++i)
    if (bvh__qselect_val(bs, i) < pivot) {
      bvh__swap(uint32_t, bs->ids[j], bs->ids[i]);
      j++;
    }
  bvh__swap(uint32_t, bs->ids[j], bs->ids[e - 1]);
  return j;
}

static void bvh__sort_qselect(bvh__BuildState *bs, uint32_t s, uint32_t e, 
                                                   uint32_t k)
{
  uint32_t p = bvh__qselect_partition(bs, s, e);
  bvh__assert(s <= k && e > k);
  if (p == k)
    return;
  else if (k < p)
    bvh__sort_qselect(bs, s, p, k);
  else
    bvh__sort_qselect(bs, p + 1, e, k);
}

static void bvh__split_half(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t nt, m;
  nt = set->end - set->start;
  m = set->start + bvh__closemult4(nt / 2);
  bvh__assert(set->start <= m && set->end > m);
  bvh__sort_qselect(bs, set->start, set->end, m);
  bvh__set4range(bs, &bs->seta, set->start, m);
  bvh__set4range(bs, &bs->setb, m, set->end);
  bvh__bbox_intersect(&bs->seta.bbox, &set->bbox);
  bvh__bbox_intersect(&bs->setb.bbox, &set->bbox);
}

/* -- Spatial splits -------------------------------------------------------- */

static uint32_t bvh__step(float f)
{
  if (f < 0.0f)
    return 0;
  else
    return 1;
}

static uint32_t bvh__splitline(float p[3], const float a[3], const float b[3],
                               uint32_t axis, float s)
{
  float ta = s - a[axis];
  float tb = s - b[axis];
  if (bvh__step(ta) ^ bvh__step(tb)) {
    float t = ta / (b[axis] - a[axis]);
    p[0] = a[0] + t * (b[0] - a[0]);
    p[1] = a[1] + t * (b[1] - a[1]);
    p[2] = a[2] + t * (b[2] - a[2]);
    return 1;
  } else
    return 0;
}

static bool bvh__splittriangle (float p[2][3], const float v0[3],
                                               const float v1[3],
                                               const float v2[3],
                                               uint32_t axis,
                                               float s)
{
  uint32_t n = 0;
  n += bvh__splitline(p[n], v0, v1, axis, s);
  n += bvh__splitline(p[n], v0, v2, axis, s);
  n += bvh__splitline(p[n], v1, v2, axis, s);
  bvh__assert((n == 0) || (n == 2));
  return n == 2;
}

static bool bvh__split_chopped(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t i = 0, j = 0;
  bvh__BBox ba = bvh__bbox_empty();
  bvh__BBox bb = bvh__bbox_empty();
  uint32_t na = 0, nb = 0;
  float ds = (set->bbox.b[1][bs->splitaxis] - set->bbox.b[0][bs->splitaxis]) /
              BVH_SAH_NUM_BUCKETS;
  float costmin = FLT_MAX;
  float smin = FLT_MAX;
  for (i = 1; i < BVH_SAH_NUM_BUCKETS; ++i) {
    /* Find position `smin' after which the cost of splitting the set of
       triangles is minimal. */
    float s = set->bbox.b[0][bs->splitaxis] + i * ds;
    float ccur = 0.0f;
    uint32_t na0 = 0, nb0 = 0;
    bvh__BBox ba0 = bvh__bbox_empty();
    bvh__BBox bb0 = bvh__bbox_empty();
    for (j = set->start; j < set->end; ++j) {
      const bvh__BBox *b = &bs->bboxes[bs->ids[j]];
      if (b->b[0][bs->splitaxis] > s) {
        bvh__bbox_union(&bb0, b);
        nb0++;
      } else if (b->b[1][bs->splitaxis] <= s) {
        bvh__bbox_union(&ba0, b);
        na0++;
      } else {
        float p[2][3];
        const float *v0, *v1, *v2;
        v0 = &bs->vertices[3 * bs->vertexids[3 * bs->ids[j] + 0]];
        v1 = &bs->vertices[3 * bs->vertexids[3 * bs->ids[j] + 1]];
        v2 = &bs->vertices[3 * bs->vertexids[3 * bs->ids[j] + 2]];
        bool issplit = bvh__splittriangle(p, v0, v1, v2, bs->splitaxis, s);
        bvh__assert(issplit);
        if (v0[bs->splitaxis] <= s)
          bvh__bbox_extend(&ba0, v0);
        else
          bvh__bbox_extend(&bb0, v0);
        if (v1[bs->splitaxis] <= s)
          bvh__bbox_extend(&ba0, v1);
        else
          bvh__bbox_extend(&bb0, v1);
        if (v2[bs->splitaxis] <= s)
          bvh__bbox_extend(&ba0, v2);
        else
          bvh__bbox_extend(&bb0, v2);
        bvh__bbox_extend(&ba0, p[0]);
        bvh__bbox_extend(&ba0, p[1]);
        bvh__bbox_extend(&bb0, p[0]);
        bvh__bbox_extend(&bb0, p[1]);
        na0++;
        nb0++;
      }
    }
    bvh__bbox_intersect(&ba0, &set->bbox); /* `ba0' and `bb0' need to be .. */
    bvh__bbox_intersect(&bb0, &set->bbox); /* .. within `set->bbox' ... */
    ccur = bvh__sah_cost(na0, &ba0, nb0, &bb0, &set->bbox);
    if (ccur < costmin) {
      costmin = ccur;
      smin = s;
      ba = ba0; bb = bb0;
      na = na0; nb = nb0;
    }
  }
  if (costmin < bs->cost) {
    /* Compute the two resulting split sets. */
    uint32_t top = bvh__ids_push(bs, na + nb);
    uint32_t i, ia, ib;
    bs->seta.start = ia = top;
    bs->setb.start = ib = top + na;
    for (i = set->start; i < set->end; ++i) {
      const uint32_t idx = bs->ids[i];
      const bvh__BBox *b = &bs->bboxes[idx];
      if (b->b[0][bs->splitaxis] > smin)
        bs->ids[ib++] = idx;
      else if (b->b[1][bs->splitaxis] <= smin)
        bs->ids[ia++] = idx;
      else {
        float ca, cb;
        bvh__BBox ba0 = ba;
        bvh__BBox bb0 = bb;
        bvh__bbox_union(&ba0, &bs->bboxes[idx]);
        bvh__bbox_union(&bb0, &bs->bboxes[idx]);
        ca = bvh__sah_cost(na, &ba0, nb - 1, &bb, &set->bbox);
        cb = bvh__sah_cost(na - 1, &ba, nb, &bb0, &set->bbox);
        if (ca < costmin && ca < cb) {
          costmin = ca;
          nb--;
          ba = ba0;
          bs->ids[ia++] = idx;
        } else if (cb < costmin) {
          costmin = cb;
          na--;
          bb = bb0;
          bs->ids[ib++] = idx;
        } else {
          bs->ids[ia++] = idx;
          bs->ids[ib++] = idx;
        }
      }
    }
    bs->seta.end = ia;
    bs->setb.end = ib;
    bs->seta.bbox = ba;
    bs->setb.bbox = bb;
    bs->cost = costmin;
    return true;
  } else
    return false;
}

/* -------------------------------------------------------------------------- */

static uint32_t bvh__splitaxis4bbox(const bvh__BBox *b)
{
  uint32_t ax = 0;
  float d[3];
  bvh__sub3(d, b->b[1], b->b[0]);
  if (d[0] < d[1]) ax = 1;
  if (d[ax] < d[2]) ax = 2;
  return ax;
}

/* Compute the bounding box of centroids for the triangles specified by `set'
   as well as the split axis along which the set of triangles should be split
   along and stores it in the build state. */
static void bvh__updatesplitinfo(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t i;
  bvh__bbox4point(&bs->cbbox, bs->centroids[bs->ids[set->start]]);
  for (i = set->start + 1; i < set->end; ++i)
    bvh__bbox_extend(&bs->cbbox, bs->centroids[bs->ids[i]]);
  bvh__bbox_intersect(&bs->cbbox, &set->bbox); /* Account for chopped triangles ... */
  bs->splitaxis = bvh__splitaxis4bbox(&bs->cbbox);
  bs->cmin = bs->cbbox.b[0][bs->splitaxis];
  bs->cmax = bs->cbbox.b[1][bs->splitaxis];
}

#define bvh__cansplit(bs) !(bvh__almeq((bs)->cmin, (bs)->cmax))

static bool bvh__needspatialsplits(const bvh__BuildState *bs)
{
  bvh__BBox u = bs->seta.bbox;
  bvh__bbox_intersect(&u, &bs->setb.bbox);
  if (bvh__bbox_area(&u) / bs->boxarea > BVH_SPATIAL_SPLIT_ALPHA)
    return true;
  else
    return false;
}

static bool bvh__split(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t n = set->end - set->start;
  bvh__assert(set->start < set->end);
  if (n <= BVH_MAX_TRIANGLES)
    return false;
  bvh__updatesplitinfo(bs, set);
  if (!bvh__cansplit(bs))
    goto splithalf;
  if (!bvh__split_default(bs, set))
    goto splithalf;
  if (bvh__needspatialsplits(bs) && bvh__split_chopped(bs, set))
    return true;
  bvh__revisesplit(bs, set);
  return true;
splithalf:
  if (n > 64) {
    bvh__split_half(bs, set);
    return true;
  } else
    return false;
}

static uint32_t bvh__makeleaf(bvh__BuildState *bs, const bvh__Set *set)
{
  uint32_t s, nt, nq, r;
  bvh__assert(set->start < set->end);

  /* Compute the quadruple count the leaf references. */
  {
    nt = set->end - set->start;
    nq = nt % 4 == 0 ? nt / 4 : (nt + 4) / 4;
    r = nq * 4 - nt;
    bvh__assert(nq <= 16);
  }

  /* Allocate memory for new bundle data in 'bvh'. */
  {
    bs->bvh->nquads += nq;
    if (bs->quadscap < bs->bvh->nquads) {
      if (!bs->quadscap)
        bs->quadscap = 1;
      while (bs->quadscap < bs->bvh->nquads)
        bs->quadscap *= 2;
      bs->bvh->trianglequads = (bvh__QTriangle *)bvh__arealloc(
          bs, bs->bvh->trianglequads, 
          bs->quadscap * sizeof(*bs->bvh->trianglequads), 16);
      bs->bvh->idquads = (bvh_id_quad_t *)bvh__arealloc(
          bs, bs->bvh->idquads,
          bs->quadscap * sizeof(*bs->bvh->idquads), sizeof(*bs->bvh->idquads));
    }
    s = bs->bvh->nquads - nq; /* Reference to the newly allocated bundles. */
  }

  /* Convert scene triangle data to a SIMD friendly format and store
     it in 'bvh'. */
  {
    const uint32_t *vids = NULL;
    uint32_t i = 0;
    for (i = 0; i < nt; ++i) {
      vids = &bs->vertexids[3 * bs->ids[set->start + i]];
      bvh__qtriangle_set(&bs->bvh->trianglequads[s + i / 4], i % 4,
                         &bs->vertices[3 * vids[0]],
                         &bs->vertices[3 * vids[1]],
                         &bs->vertices[3 * vids[2]]);
      bs->bvh->idquads[s + i / 4][i % 4] = bs->ids[set->start + i];
    }

    /* Copy the last triangle of the set to be referenced by the leaf node
       until the number of triangles initialized is a multiple of four. */
    vids = &bs->vertexids[3 * bs->ids[set->end - 1]];
    for (i = nt; i < nt + r; ++i) {
      bvh__qtriangle_set(&bs->bvh->trianglequads[s + i / 4], i % 4,
                         &bs->vertices[3 * vids[0]],
                         &bs->vertices[3 * vids[1]],
                         &bs->vertices[3 * vids[2]]);
      bs->bvh->idquads[s + i / 4][i % 4] = bs->ids[set->end - 1];
    }
  }
  return bvh__leaf_encodeid(s, nq);
}

static uint32_t bvh__pushnode(bvh__BuildState *bs, const bvh__Node *node)
{
  if (bs->nodescap == bs->bvh->nnodes) {
    if (!bs->nodescap)
      bs->nodescap = 1;
    bs->nodescap *= 2;
    bs->bvh->nodes = (bvh__Node *)bvh__arealloc(
                          bs, bs->bvh->nodes, 
                          bs->nodescap * sizeof(*bs->bvh->nodes),
                          sizeof(*bs->bvh->nodes));
  }
  bs->bvh->nodes[bs->bvh->nnodes] = *node;
  return bs->bvh->nnodes++;
}

static uint32_t bvh__buildnode(bvh__BuildState *bs, const bvh__Set *set)
{
  if (bvh__split(bs, set)) {
    bvh__Node node = {0};
    bvh__SplitResult res = bvh__splitres(bs);
    node.axis[0] = res.splitaxis; 
    if (bvh__split(bs, &res.seta)) {
      bvh__SplitResult res = bvh__splitres(bs);
      node.axis[1] = res.splitaxis;
      node.childids[0] = bvh__buildnode(bs, &res.seta);
      node.childids[1] = bvh__buildnode(bs, &res.setb);
      bvh__qbbox_set(&node.childbboxes, 0, &res.seta.bbox);
      bvh__qbbox_set(&node.childbboxes, 1, &res.setb.bbox);
    } else {
      node.axis[1] = 0;
      node.childids[0] = bvh__makeleaf(bs, &res.seta);
      node.childids[1] = BVH__NODE_EMPTY;
      bvh__qbbox_set(&node.childbboxes, 0, &res.seta.bbox);
      bvh__qbbox_setzero(&node.childbboxes, 1);
    }
    if (bvh__split(bs, &res.setb)) {
      bvh__SplitResult res = bvh__splitres(bs);
      node.axis[2] = res.splitaxis;
      node.childids[2] = bvh__buildnode(bs, &res.seta);
      node.childids[3] = bvh__buildnode(bs, &res.setb);
      bvh__qbbox_set(&node.childbboxes, 2, &res.seta.bbox);
      bvh__qbbox_set(&node.childbboxes, 3, &res.setb.bbox);
    } else {
      node.axis[2] = 0;
      node.childids[2] = bvh__makeleaf(bs, &res.setb);
      node.childids[3] = BVH__NODE_EMPTY;
      bvh__qbbox_set(&node.childbboxes, 2, &res.setb.bbox);
      bvh__qbbox_setzero(&node.childbboxes, 3);
    }
    return bvh__pushnode(bs, &node);
  } else
    return bvh__makeleaf(bs, set);
}

static void bvh__initbs(bvh__BuildState *bs, const uint32_t vids[],
                                             const float v[],
                                             const uint32_t n)
{
  uint32_t i = 0;
  bs->bboxes = (bvh__BBox *)bvh__amalloc(bs, n * sizeof(*bs->bboxes), 4);
  bs->centroids = (bvh__Vec3 *)bvh__amalloc(bs, n * sizeof(*bs->centroids), 4);
  bvh__ids_push(bs, n);
  for (i = 0; i < n; ++i) {
    bs->ids[i] = i;
    bvh__bbox4triangle(&bs->bboxes[i], &v[3 * vids[3 * i + 0]],
                                       &v[3 * vids[3 * i + 1]],
                                       &v[3 * vids[3 * i + 2]]);
    bvh__bbox_center(bs->centroids[i], &bs->bboxes[i]);
  }
  bs->vertexids = vids;
  bs->vertices = v;
  bs->ntriangles = n;
  bs->bvh = (bvh__Data *)bvh__amalloc(bs, sizeof(*bs->bvh), 4);
  memset(bs->bvh, 0, sizeof(*bs->bvh));
  memcpy(&bs->bvh->bounds, &bs->bboxes[0], sizeof(bs->bboxes[0]));
  for (i = 1; i < n; ++i)
    bvh__bbox_union(&bs->bvh->bounds, &bs->bboxes[i]);
  bs->boxarea = bvh__bbox_area(&bs->bvh->bounds);
}

bvh_Handle bvh_build(const uint32_t vertexids[],
                     const float vertices[],
                     uint32_t ntriangles)
{
  bvh__BuildState bs = {0};
  BVH__TRY(&bs)
    bvh__Set set = {0};
    bvh__initbs(&bs, vertexids, vertices, ntriangles);
    set.start = 0;
    set.end = ntriangles;
    set.bbox = bs.bvh->bounds;
    bs.bvh->root = bvh__buildnode(&bs, &set);
  BVH__CATCH(&bs)
    bvh_free(&bs.bvh);
  BVH__END(&bs)
  bvh__afree(bs.bboxes);
  bvh__afree(bs.centroids);
  bvh__afree(bs.ids);
  return bs.bvh;
}

void bvh_free(bvh_Handle *bvh)
{
  if (!bvh || !(*bvh))
    return;
  bvh__afree((*bvh)->nodes);
  bvh__afree((*bvh)->trianglequads);
  bvh__afree((*bvh)->idquads);
  bvh__afree(*bvh);
  *bvh = NULL;
}

/* -- Tracing rays ---------------------------------------------------------- */

typedef struct {
  __m128 ox;
  __m128 oy;
  __m128 oz;
  __m128 dx;
  __m128 dy;
  __m128 dz;
  __m128 idx;
  __m128 idy;
  __m128 idz;
  uint32_t nd[3]; /* Is direction negative? */
  __m128 tmin;
  __m128 tmax;
} bvh__QRay;

static void bvh_setray(bvh__QRay *ray, const float o[3], const float d[3],
                                             float tmin, float tmax)
{
  ray->ox = _mm_set_ps1(o[0]);
  ray->oy = _mm_set_ps1(o[1]);
  ray->oz = _mm_set_ps1(o[2]);
  ray->dx = _mm_set_ps1(d[0]);
  ray->dy = _mm_set_ps1(d[1]);
  ray->dz = _mm_set_ps1(d[2]);
  ray->idx = _mm_set_ps1(1.0f / d[0]);
  ray->idy = _mm_set_ps1(1.0f / d[1]);
  ray->idz = _mm_set_ps1(1.0f / d[2]);
  ray->nd[0] = d[0] < 0.0f;
  ray->nd[1] = d[1] < 0.0f;
  ray->nd[2] = d[2] < 0.0f;
  ray->tmin = _mm_set_ps1(tmin);
  ray->tmax = _mm_set_ps1(tmax);
}

static __m128 bvh__intersect_ray_bbox(const bvh__QRay *r, const bvh__QBBox *b)
{
  __m128 tmin, tmax, tymin, tymax, tzmin, tzmax, res;
  tmin = _mm_mul_ps(_mm_sub_ps(b->bx[r->nd[0]], r->ox), r->idx);
  tmax = _mm_mul_ps(_mm_sub_ps(b->bx[1 - r->nd[0]], r->ox), r->idx);
  tymin = _mm_mul_ps(_mm_sub_ps(b->by[r->nd[1]], r->oy), r->idy);
  tymax = _mm_mul_ps(_mm_sub_ps(b->by[1 - r->nd[1]], r->oy), r->idy);
  res = _mm_and_ps(_mm_cmple_ps(tmin, tymax), _mm_cmple_ps(tymin, tmax));
  tmin = _mm_max_ps(tmin, tymin);
  tmax = _mm_min_ps(tmax, tymax);
  tzmin = _mm_mul_ps(_mm_sub_ps(b->bz[r->nd[2]], r->oz), r->idz);
  tzmax = _mm_mul_ps(_mm_sub_ps(b->bz[1 - r->nd[2]], r->oz), r->idz);
  res = _mm_and_ps(res, _mm_and_ps(_mm_cmple_ps(tmin, tzmax),
                                   _mm_cmple_ps(tzmin, tmax)));
  tmin = _mm_max_ps(tmin, tzmin);
  tmax = _mm_min_ps(tmax, tzmax);
  res = _mm_and_ps(res, _mm_and_ps(_mm_cmplt_ps(tmin, r->tmax),
                                   _mm_cmpgt_ps(tmax, r->tmin)));
  return res;
}

static void bvh__intersect_ray_triangle(__m128 *t, __m128 *u, __m128 *v,
                                        const bvh__QRay *ray,
                                        const bvh__QTriangle *tr)
{
  __m128 e1x, e1y, e1z, e2x, e2y, e2z;
  __m128 px, py, pz, qx, qy, qz;
  __m128 tx, ty, tz;
  __m128 det, idet;
  __m128 eps, one, zero;
  __m128 res;

  eps = _mm_set_ps1(0.000001);
  one = _mm_set_ps1(1.0);
  zero = _mm_set_ps1(0.0);

  e1x = _mm_sub_ps(tr->v1x, tr->v0x);
  e1y = _mm_sub_ps(tr->v1y, tr->v0y);
  e1z = _mm_sub_ps(tr->v1z, tr->v0z);

  e2x = _mm_sub_ps(tr->v2x, tr->v0x);
  e2y = _mm_sub_ps(tr->v2y, tr->v0y);
  e2z = _mm_sub_ps(tr->v2z, tr->v0z);

  px = _mm_sub_ps(_mm_mul_ps(ray->dy, e2z), _mm_mul_ps(ray->dz, e2y));
  py = _mm_sub_ps(_mm_mul_ps(ray->dz, e2x), _mm_mul_ps(ray->dx, e2z));
  pz = _mm_sub_ps(_mm_mul_ps(ray->dx, e2y), _mm_mul_ps(ray->dy, e2x));

  det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, px), _mm_mul_ps(e1y, py)),
                                                   _mm_mul_ps(e1z, pz));

  res = _mm_or_ps(_mm_cmple_ps(det, eps), _mm_cmpge_ps(det, eps));

  idet = _mm_div_ps(one, det);

  tx = _mm_sub_ps(ray->ox, tr->v0x);
  ty = _mm_sub_ps(ray->oy, tr->v0y);
  tz = _mm_sub_ps(ray->oz, tr->v0z);

  *u = _mm_mul_ps(idet, _mm_add_ps(_mm_add_ps(_mm_mul_ps(tx, px),
                                              _mm_mul_ps(ty, py)),
                                              _mm_mul_ps(tz, pz)));

  res = _mm_and_ps(res, _mm_and_ps(_mm_cmpge_ps(*u, zero),
                                   _mm_cmplt_ps(*u, one)));

  qx = _mm_sub_ps(_mm_mul_ps(ty, e1z), _mm_mul_ps(tz, e1y));
  qy = _mm_sub_ps(_mm_mul_ps(tz, e1x), _mm_mul_ps(tx, e1z));
  qz = _mm_sub_ps(_mm_mul_ps(tx, e1y), _mm_mul_ps(ty, e1x));

  *v = _mm_mul_ps(idet, _mm_add_ps(_mm_add_ps(_mm_mul_ps(ray->dx, qx),
                                              _mm_mul_ps(ray->dy, qy)),
                                              _mm_mul_ps(ray->dz, qz)));

  res = _mm_and_ps(res, _mm_and_ps(_mm_cmpge_ps(*v, zero),
                        _mm_cmple_ps(_mm_add_ps(*u, *v), one)));

  *t = _mm_mul_ps(idet, _mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qx),
                                              _mm_mul_ps(e2y, qy)),
                                              _mm_mul_ps(e2z, qz)));
  res = _mm_and_ps(res, _mm_and_ps(_mm_cmplt_ps(*t, ray->tmax),
                                   _mm_cmpgt_ps(*t, ray->tmin)));
  *t = _mm_or_ps(_mm_and_ps(res, *t), _mm_andnot_ps(res, _mm_set_ps1(INFINITY)));
}

static bool bvh__intersect_leaf(const bvh__Data *bvh, uint32_t leafid,
                                const bvh__QRay *ray, float *t,
                                float *u, float *v, uint32_t *faceid)
{
  uint32_t i, nq, st;
  __m128 t0, u0, v0;
  __m128i id0;
  nq = bvh__leaf_nquads(leafid);
  st = bvh__leaf_quadid(leafid);
  bvh__assert(nq);

  /* Trace all quads referenced by the leaf node. */
  {
    bvh__intersect_ray_triangle(&t0, &u0, &v0, ray, &bvh->trianglequads[st]);
    id0 = _mm_set_epi32(bvh->idquads[st][3], bvh->idquads[st][2],
        bvh->idquads[st][1], bvh->idquads[st][0]);
    for (i = 1; i < nq; ++i) {
      __m128 t1, u1, v1, m;
      __m128i id1 = _mm_set_epi32(bvh->idquads[st + i][3], bvh->idquads[st + i][2],
          bvh->idquads[st + i][1], bvh->idquads[st + i][0]);
      bvh__intersect_ray_triangle(&t1, &u1, &v1, ray, &bvh->trianglequads[st + i]);

      /* For each quad slot: store the results of the closest intersection. */
      m = _mm_cmplt_ps(t1, t0);
      t0 = _mm_or_ps(_mm_and_ps(m, t1), _mm_andnot_ps(m, t0));
      u0 = _mm_or_ps(_mm_and_ps(m, u1), _mm_andnot_ps(m, u0));
      v0 = _mm_or_ps(_mm_and_ps(m, v1), _mm_andnot_ps(m, v0));
      id0 = _mm_or_si128(_mm_and_si128(_mm_castps_si128(m), id1), _mm_andnot_si128(_mm_castps_si128(m), id0));
    }
  }

  /* Get the result for the closest intersection and return. */
  {
    uint32_t i = 0;
    float *t2 = (float *)&t0;
    if (t2[1] < t2[0]) i = 1;
    if (t2[2] < t2[i]) i = 2;
    if (t2[3] < t2[i]) i = 3;
    if (t2[i] == INFINITY) return false;
    *t = t2[i];
    *u = ((float *)&u0)[i];
    *v = ((float *)&v0)[i];
    *faceid = ((uint32_t *)&id0)[i];
    return true;
  }
}

bool bvh_trace(const bvh_Handle bvh, float *t, float *u, float *v,
               uint32_t *faceid, const float o[3], const float d[3],
               float tmin, float tmax)
{
  int32_t top = 64 - 1;
  uint32_t todos[64]; /* Too big for stack? */
  uint32_t id;
  float t0, u0, v0;
  bvh__QRay r;
  todos[top--] = bvh->root;
  bvh_setray(&r, o, d, tmin, tmax);
  *t = FLT_MAX;
  while (top != (64 - 1)) {
    const bvh__Node *node;
    uint32_t nid = todos[++top];
    node = &bvh->nodes[nid];
    if (bvh_isempty(nid))
      continue;
    else if (bvh_isleaf(nid)) {
      if (bvh__intersect_leaf(bvh, nid, &r, &t0, &u0, &v0, &id)) {
        if (t0 < *t) {
          r.tmax = _mm_set_ps1(t0);
          *t = t0;
          *u = u0;
          *v = v0;
          *faceid = id;
        }
      }
    } else {
      __m128i in, ch;
      int32_t *oin;
      uint32_t *och;
      uint32_t mask;
      bvh__assert(top <= 64 && top > 4);
      in = _mm_castps_si128(bvh__intersect_ray_bbox(&r, &bvh->nodes[nid].childbboxes));
      ch = _mm_set_epi32(node->childids[3], node->childids[2],
          node->childids[1], node->childids[0]);
      /* Re-order the child references for more efficient ray traversal.
         See [1] section 2.3 for details. */
      mask = ((1 - r.nd[node->axis[1]]) << 0) |
             ((1 - r.nd[node->axis[0]]) << 1) |
             ((    r.nd[node->axis[1]]) << 2) |
             ((1 - r.nd[node->axis[0]]) << 3) |
             ((1 - r.nd[node->axis[2]]) << 4) |
             ((    r.nd[node->axis[0]]) << 5) |
             ((    r.nd[node->axis[2]]) << 6) |
             ((    r.nd[node->axis[0]]) << 7);
      switch (mask) {
        case 0xe4:
          ch = _mm_shuffle_epi32(ch, 0xe4);
          in = _mm_shuffle_epi32(in, 0xe4);
          break;
        case 0xb4:
          ch = _mm_shuffle_epi32(ch, 0xb4);
          in = _mm_shuffle_epi32(in, 0xb4);
          break;
        case 0xe1:
          ch = _mm_shuffle_epi32(ch, 0xe1);
          in = _mm_shuffle_epi32(in, 0xe1);
          break;
        case 0xb1:
          ch = _mm_shuffle_epi32(ch, 0xb1);
          in = _mm_shuffle_epi32(in, 0xb1);
          break;
        case 0x4e:
          ch = _mm_shuffle_epi32(ch, 0x4e);
          in = _mm_shuffle_epi32(in, 0x4e);
          break;
        case 0x1e:
          ch = _mm_shuffle_epi32(ch, 0x1e);
          in = _mm_shuffle_epi32(in, 0x1e);
          break;
        case 0x4b:
          ch = _mm_shuffle_epi32(ch, 0x4b);
          in = _mm_shuffle_epi32(in, 0x4b);
          break;
        case 0x1b:
          ch = _mm_shuffle_epi32(ch, 0x1b);
          in = _mm_shuffle_epi32(in, 0x1b);
          break;
        default:
          bvh__assert(false);
      }
      oin = (int32_t *)&in;
      och = (uint32_t *)&ch;
      todos[top] = och[0]; top += oin[0];
      todos[top] = och[1]; top += oin[1];
      todos[top] = och[2]; top += oin[2];
      todos[top] = och[3]; top += oin[3];
    }
  }
  if (*t == FLT_MAX)
    return false;
  else
    return true;
}

/* -------------------------------------------------------------------------- */

#endif
