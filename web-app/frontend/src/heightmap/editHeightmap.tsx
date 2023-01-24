import changeTexture2D from "../common/changeTexture2D";

export default function editHeightmap(params: any, mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture, loader: THREE.TextureLoader, mapsArray: Array<Array<string>>, mapsContainer: HTMLElement, heightmap2d: HTMLImageElement, texture2d: HTMLImageElement, ctx: CanvasRenderingContext2D, worldWidth: number, worldDepth: number) {
    const editHeightmapCanvas = document.querySelector('#edit_heightmap_canvas') as HTMLCanvasElement;
    editHeightmapCanvas.height = 512;
    editHeightmapCanvas.width = 512;
    const editHeightmapCtx = editHeightmapCanvas.getContext('2d') as CanvasRenderingContext2D;
    const uploader = document.querySelector('#btn_edit_heightmap') as HTMLButtonElement;
    uploader.addEventListener('click', (e) => {
        const editHeightmapContainer = document.querySelector('#edit_heightmap_container') as HTMLDivElement;
        editHeightmapContainer.style.display = 'flex';

        const img = new Image();
        img.src = texture2d.src;
        img.onload = function () {
            console.log('image uploaded');
            console.log(img.height, img.width);
            editHeightmapCtx.drawImage(img, 0, 0, editHeightmapCanvas.height, editHeightmapCanvas.width);
        };

        let painting = false;

        function startPosition(e: MouseEvent) {
            console.log('START');
            painting = true;
            draw(e);
        }

        function finishedPosition() {
            painting = false;
            editHeightmapCtx.beginPath();
        }

        function draw(e: MouseEvent) {
            if (!painting) return;
            const x = editHeightmapCanvas?.getBoundingClientRect().x
            const y = editHeightmapCanvas?.getBoundingClientRect().y;
            editHeightmapCtx.lineWidth = params.lineWidth;
            editHeightmapCtx.strokeStyle = 'rgba(255,255,255,255)';
            editHeightmapCtx.lineCap = "round";

            editHeightmapCtx.lineTo(e.clientX - x, e.clientY - y);
            editHeightmapCtx.stroke();
            editHeightmapCtx.beginPath();
            editHeightmapCtx.moveTo(e.clientX - x, e.clientY - y);
        }

        editHeightmapCanvas.addEventListener("mousedown", startPosition);
        document.addEventListener("mouseup", finishedPosition);
        document.addEventListener("mousemove", draw);
    })

    const btnExitEditHeightmap = document.querySelector('#btn_exit_edit_heightmap') as HTMLButtonElement;
    btnExitEditHeightmap.addEventListener('click', (e) => {
        const editHeightmapContainer = document.querySelector('#edit_heightmap_container') as HTMLDivElement;
        editHeightmapContainer.style.display = 'none';
    }
    );

    const fillTexture = () => {
        if (texture2d.src.slice(0, 22) !== 'data:image/png;base64,') {
            throw Error('No texture image src.')
        }
        const imageData = editHeightmapCtx.getImageData(0, 0, editHeightmapCanvas.width, editHeightmapCanvas.height);
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 256;
        tempCanvas.height = 256;
        const tempCtx = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
        tempCtx.putImageData(imageData, 0, 0, 0, 0, 512, 512);
        tempCtx.drawImage(editHeightmapCanvas, 0, 0, 512, 512, 0, 0, 256, 256);
        const img = new Image();
        img.src = tempCanvas.toDataURL("image/jpeg", 1.0);
        
        console.log('SRC', editHeightmapCtx.getImageData(0, 0, 10, 10));
        fetch('http://127.0.0.1:8000/inpaiting', {
            method: 'POST',
            body: JSON.stringify({ 'image64': img.src.slice(22), 'dataset': params.currentDataset }),
            headers: {
                'Content-Type': 'application/json'
            },
        }).then(response => {
            console.log(response);
            response.json().then(res => {
                changeTexture2D(res.image64, params, mesh, texture, loader, mapsArray, mapsContainer, texture2d);
            })
        })
    }
    const buttonFillTexture = document.querySelector('#btn_fill_texture') as HTMLButtonElement;
    buttonFillTexture.addEventListener('click', fillTexture);
}