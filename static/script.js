// --- UTILITIES ---
const showMessage = (text, type = 'info') => {
    const container = document.getElementById('globalMessageContainer');
    if (!container) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = text;
    container.appendChild(messageDiv);
    setTimeout(() => {
        messageDiv.style.opacity = '0';
        setTimeout(() => messageDiv.remove(), 500);
    }, 3000);
};

const showLoadingAndReload = () => {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.innerHTML = '<div class="loader"></div>';
        document.body.appendChild(overlay);
    }
    setTimeout(() => {
        overlay.classList.add('is-active');
    }, 10);

    setTimeout(() => {
        window.location.reload();
    }, 300);
};

const navigateWithLoader = (url) => {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.innerHTML = '<div class="loader"></div>';
        document.body.appendChild(overlay);
    }
    setTimeout(() => {
        overlay.classList.add('is-active');
    }, 10);

    setTimeout(() => {
        window.location.href = url;
    }, 300);
};


const showConfirmation = (title, text, onConfirm) => {
    const container = document.getElementById('confirmationModal');
    if (!container) return;
    container.innerHTML = `
        <div class="confirmation-modal-content">
            <h3>${title}</h3>
            <p>${text}</p>
            <div class="modal-buttons">
                <button id="confirmBtn" class="primary-btn">Yes</button>
                <button id="cancelBtn" class="reset-btn">No</button>
            </div>
        </div>`;
    container.style.display = 'flex';
    container.querySelector('#confirmBtn').onclick = () => { onConfirm(); container.style.display = 'none'; };
    container.querySelector('#cancelBtn').onclick = () => { container.style.display = 'none'; };
};

// --- CART RENDERING ---
const renderCartModal = (cartData) => {
    const header = document.getElementById('headerCartSummary');
    if (header) header.innerHTML = `<i class="fas fa-shopping-cart"></i>&nbsp; Cart (${cartData.total_items})`;

    const cartModal = document.getElementById('cartModal');
    if (!cartModal) return;

    const modalTableBody = cartModal.querySelector('tbody');
    const modalMessage = cartModal.querySelector('#modalCartMessage');
    const modalTable = cartModal.querySelector('#modalCartTable');

    if (!modalTableBody || !modalMessage || !modalTable) return;

    modalTableBody.innerHTML = '';
    if (cartData.cart_items.length === 0) {
        modalMessage.style.display = 'block';
        modalTable.style.display = 'none';
    } else {
        modalMessage.style.display = 'none';
        modalTable.style.display = 'table';
        cartData.cart_items.forEach(item => {
            const row = modalTableBody.insertRow();
            const categoryText = encodeURIComponent(item.category);
            row.innerHTML = `
                <td class="modal-product-cell">
                    <img src="https://placehold.co/60x60/f7f2f0/666?text=${categoryText}" alt="${item.product_name}">
                    <div>
                        <a href="/product/${item.id}">${item.product_name}</a>
                        <span>${item.category}</span>
                    </div>
                </td>
                <td class="modal-product-price">$${item.price_per_item.toFixed(2)}</td>
                <td>${item.quantity}</td>
                <td class="modal-product-subtotal">$${item.subtotal.toFixed(2)}</td>
                <td>
                    <button class="remove-item-btn" onclick="confirmRemoveFromCart('${item.id}')">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </td>`;
        });
    }
};

// --- API CALLS ---
const apiCall = async (endpoint, options = {}) => {
    try {
        const resp = await fetch(`/api/${endpoint}`, options);
        if (!resp.ok) {
            let errorMsg = resp.statusText;
            try { const errorData = await resp.json(); errorMsg = errorData.error || errorMsg; }
            catch (e) { /* Ignore */ }
            throw new Error(errorMsg);
        }
        return resp.json();
    } catch (err) {
        showMessage(err.message, 'error');
        throw err;
    }
};

// --- GLOBALLY ACCESSIBLE FUNCTIONS ---
window.openCartModal = async () => {
    const modal = document.getElementById('cartModal');
    if (modal) {
        modal.style.display = 'flex';
        try {
            const cartData = await apiCall('get_cart');
            renderCartModal(cartData);
        } catch (error) {
            console.error("Failed to fetch cart data for modal.", error);
        }
    }
};

window.closeCartModal = () => {
    document.getElementById('cartModal').style.display = 'none';
};

window.addToCartSimple = (productId) => _addToCart(productId, 1);

window.addToCartFromDetail = (productId) => {
    const quantityInput = document.getElementById('quantity');
    const quantity = quantityInput ? parseInt(quantityInput.value, 10) : 1;
    _addToCart(productId, quantity);
};

const _addToCart = async (productId, quantity) => {
    try {
        await apiCall(`add_to_cart/${productId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ quantity: quantity })
        });
        showLoadingAndReload();
    } catch (error) {}
};

window.confirmRemoveFromCart = (productId) => {
    showConfirmation('Remove Item?', 'Are you sure?', () => _removeFromCart(productId));
};

const _removeFromCart = async (productId) => {
    try {
        await apiCall(`remove_from_cart/${productId}`, { method: 'POST' });
        showLoadingAndReload();
    } catch (error) {}
};

window.updateTableQuantity = async (productId, newQty) => {
    if (newQty < 1) {
        confirmRemoveFromCart(productId);
        return;
    }
    try {
        await apiCall('update_cart_quantity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_id: productId, quantity: newQty })
        });
        showLoadingAndReload();
    } catch (error) {}
};

window.confirmResetCart = () => {
    showConfirmation('Clear Basket?', 'Are you sure?', async () => {
        try {
            await apiCall('reset_cart', { method: 'POST' });
            showLoadingAndReload();
        } catch (error) {}
    });
};

window.changeMainImage = (thumbElement) => {
    const mainImage = document.getElementById('mainImage');
    if (mainImage) mainImage.src = thumbElement.src.replace('/100/100', '/600/600');
    document.querySelectorAll('.product-thumbnails img').forEach(thumb => thumb.classList.remove('active'));
    thumbElement.classList.add('active');
};

window.changeQuantity = (amount) => {
    const quantityInput = document.getElementById('quantity');
    if (quantityInput) {
        let currentVal = parseInt(quantityInput.value, 10);
        if (currentVal + amount >= 1) {
            quantityInput.value = currentVal + amount;
        }
    }
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    const cartModal = document.getElementById('cartModal');
    const confirmationModal = document.getElementById('confirmationModal');
    cartModal?.addEventListener('click', (e) => { if (e.target === cartModal) closeCartModal(); });
    confirmationModal?.addEventListener('click', (e) => { if (e.target === confirmationModal) e.target.style.display = 'none'; });

    // Add navigation loader to category links
    document.querySelectorAll('a.nav-category-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault(); // Stop the browser from navigating instantly
            navigateWithLoader(this.href);
        });
    });

    // Add navigation loader to sort dropdown
    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', (event) => {
            const selectedSort = event.target.value;
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('sort', selectedSort);
            navigateWithLoader(currentUrl.href);
        });
    }
});
