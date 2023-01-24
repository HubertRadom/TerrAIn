import * as THREE from 'three';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import chooseDataset from './dataset/chooseDataset';
import generateHeightmapHandler from './dataset/generateHeightmap';
import importHeightmapHandler from './dataset/importHeightmap';
import colourHeightmapHandler from './heightmap/colourHeightmap';
import editHeightmap from './heightmap/editHeightmap';
import exportHandler from './heightmap/exportHeightmap';

let container, stats: { dom: any; update: () => void; };

let camera: THREE.PerspectiveCamera, controls, scene: THREE.Scene, renderer: THREE.WebGLRenderer;

let mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture;

const worldWidth = 256, worldDepth = 256,
    worldHalfWidth = worldWidth / 2, worldHalfDepth = worldDepth / 2;

let helper: THREE.Mesh<THREE.ConeGeometry, THREE.MeshNormalMaterial>;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const loader = new THREE.TextureLoader();

const heightmap2d = document.querySelector('#heightmap2d img') as HTMLImageElement;
heightmap2d.src = './heightmaps/heightmap0.png';

const texture2d = document.querySelector('#texture2d img') as HTMLImageElement;

const mapsContainer = document.querySelector('#maps_container') as HTMLDivElement;
const mapsArray: string[][] = [];
const heightmapsArray: Float64Array[] = [];

export type paramsType = {
    currentDataset: 'death_valley' | 'laytonville' | 'san_gabriel' | 'post_earthquake' | 'mt_rainier',
    currentId: number,
    lineWidth: number,
    heightmap: Float64Array[],
};

const params: paramsType = {
    currentDataset: 'death_valley',
    currentId: -1,
    lineWidth: 10,
    heightmap: [],
}
let btnCurrentDataset = document.querySelector('#dataset_death_valley') as HTMLButtonElement;

const infoDiv = document.querySelector('#info_container > div') as HTMLDivElement;

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

container = document.getElementById('container') as HTMLElement;
container.innerHTML = '';

renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth * 0.5, window.innerHeight);
container.appendChild(renderer.domElement);

scene = new THREE.Scene();
scene.background = new THREE.Color(0xbfd1e5);

camera = new THREE.PerspectiveCamera(80, window.innerWidth * 0.5 / window.innerHeight, 10, 20000);

controls = new OrbitControls(camera, renderer.domElement);
controls.minDistance = 1000;
controls.maxDistance = 15000;
controls.maxPolarAngle = Math.PI / 2;

camera.position.y = 8000;
camera.position.x = 8000;
controls.update();

const geometry = new THREE.PlaneGeometry(7500, 7500, worldWidth - 1, worldDepth - 1);
geometry.rotateX(- Math.PI / 2);

texture = loader.load('./heightmaps/heightmap0.png');

mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: texture }));
mesh.name = 'heightmap';

scene.add(mesh);

heightmap2d.addEventListener('click', () => {
    texture = loader.load(heightmap2d.src);
    mesh.material.map = texture;
})

texture2d.addEventListener('click', () => {
    texture = loader.load(texture2d.src);
    mesh.material.map = texture;
})

const geometryHelper = new THREE.ConeGeometry(20, 100, 3);
geometryHelper.translate(0, 50, 0);
geometryHelper.rotateX(Math.PI / 2);
helper = new THREE.Mesh(geometryHelper, new THREE.MeshNormalMaterial());
scene.add(helper);

container.addEventListener('pointermove', onPointerMove);

stats = Stats();
container.appendChild(stats.dom);

window.addEventListener('resize', onWindowResize);

const btnChooseDataset = document.querySelector('#btn_choose_dataset') as HTMLButtonElement;
btnChooseDataset.addEventListener('click', (e) => {
    const chooseDatasetContainer = document.querySelector('#choose_dataset_container') as HTMLDivElement;
    chooseDatasetContainer.style.display = 'flex';
});

const btnExitChooseDataset = document.querySelector('#btn_exit_choose_dataset') as HTMLButtonElement;
btnExitChooseDataset.addEventListener('click', (e) => {
    const chooseDatasetContainer = document.querySelector('#choose_dataset_container') as HTMLDivElement;
    chooseDatasetContainer.style.display = 'none';
});

const inputRange = document.querySelector('#range_edit_heightmap') as HTMLInputElement;
inputRange.addEventListener('input', (e) => {
    const value = Number(inputRange.value);
    params.lineWidth = value;
});    

document.querySelectorAll('.anchor').forEach(x => x.addEventListener('mouseover', (event) => {
    infoDiv.innerText = x.id;
    infoDiv.style.display = 'block';
    switch(x.id) {
        case 'btn_import_heightmap':
            infoDiv.innerText = 'upload heightmap image';
            break;
        case 'btn_generate_heightmap':
            infoDiv.innerText = 'generate heightmap';
            break;
        case 'btn_edit_heightmap':
            infoDiv.innerText = 'edit heightmap';
            break;
        case 'btn_fill_texture':
            infoDiv.innerText = 'fill texture';
            break;
        case 'btn_exit_edit_heightmap':
            infoDiv.innerText = 'exit edit heightmap';
            break;
        case 'btn_exit_edit_heightmap':
            infoDiv.innerText = 'colour heightmap';
            break;
        case 'range_edit_heightmap':
            infoDiv.innerText = 'edit pencil size';
            break;
    }
}));

document.querySelectorAll('.anchor').forEach(x => x.addEventListener('mouseout', (event) => { 
    infoDiv.style.display = 'none';
}));

animate();
importHeightmapHandler(params, mesh, texture, loader, mapsArray, heightmapsArray, mapsContainer, heightmap2d, texture2d, ctx, worldWidth, worldDepth);
colourHeightmapHandler(params, mesh, texture, loader, mapsArray, heightmapsArray, mapsContainer, heightmap2d, texture2d);
generateHeightmapHandler(params, mesh, texture, loader, mapsArray, heightmapsArray, mapsContainer, heightmap2d, texture2d, ctx, worldWidth, worldDepth);
chooseDataset(params, btnCurrentDataset);
editHeightmap(params, mesh, texture, loader, mapsArray, mapsContainer, heightmap2d, texture2d, ctx, worldWidth, worldDepth);
exportHandler(params, heightmapsArray, texture2d)

function onWindowResize() {

    camera.aspect = window.innerWidth * 0.5 / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth * 0.5, window.innerHeight);

}

function animate() {

    requestAnimationFrame(animate);

    render();
    stats.update();

}

function render() {

    renderer.render(scene, camera);

}

function onPointerMove(event: { clientX: number; clientY: number; }) {
    pointer.x = ((event.clientX-window.innerWidth*0.25) / renderer.domElement.clientWidth) * 2 - 1;
    pointer.y = - (event.clientY / renderer.domElement.clientHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);

    // See if the ray from the camera into the world hits one of our meshes
    const intersects = raycaster.intersectObject(mesh);

    // Toggle rotation bool for meshes that we clicked
    if (intersects.length > 0) {

        helper.position.set(0, 0, 0);

        if (!intersects[0].face) {
            throw Error('onPointerMove error');
        }
        helper.lookAt(intersects[0].face.normal);

        helper.position.copy(intersects[0].point);

    }

}